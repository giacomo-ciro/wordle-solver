from random import random
import yaml
from rich.console import Console
from collections import Counter
import numpy as np
from itertools import product

# ALGORITHM DETAILS:
#
# 1. Use a trie data structure to store the word list for faster pruning
#
# 2. Compute the opening word once as follows:
#   2.1 Compute letters' position-wise frequency
#   2.2 Generate all possible combination of these letters without repetition
#   2.3 Select the one maximizing the entropy (using the pattern distribution on all the words from the word list)
#
# 3. From the second guess:
#   3.1 If more than 50 words remain, generate a word as above (a "new" opening word)
#   3.2 Otherwise, compute the entropy of the remaining words
#       3.2.1 If the entropy is close enough to the theoretical maximum ( < 0.5 difference), use this as the guess
#       3.2.2 Otherwise, generate a word as above (a "new" opening word)
# 
# 4. Whenever less than 5 unique ambiguous letters are left in the word list (e.g., hound, bound, found):
#       4.1 Generate a shortcut word made of a random combination of these letters without repetition
#       4.2 Add it to the candidate guesses for this round (i.e., it will be selected if entropy is high enough)  


class Trie:
    def __init__(self):
        self.root = {}
        self.list = []

    def insert(self, word):
        cur = self.root
        for char in word:
            if char not in cur:
                cur[char] = {}
            cur = cur[char]
        cur["eow"] = True
        self.list.append(word)

    def filter_words(self, gray, gray_pos, yellow, green_pos, guess):
        """
        Filters the original trie and returns a copy.
        """
        new_trie = Trie()

        # Pre-compute set versions of collections for faster lookups
        gray_set = set(gray)
        green_values = set(green_pos.values())
        
        def dfs(cur, current_word, yellow, new_node):
            # Early termination for impossible words
            current_pos = len(current_word) - 1
            
            # Green check
            if current_pos in green_pos and current_word[-1] != green_pos[current_pos]:
                return False

            # Gray check
            if current_pos in gray_pos and current_word[-1] == gray_pos[current_pos]:
                return False

            # Found a word
            if "eow" in cur:
                # Skip the guess word
                if current_word == guess:
                    return False
                
                # Check yellow conditions
                for char, count in yellow.items():
                    if count > 0 and current_word.count(char) < count:
                        return False
                
                # Word passes all filters
                new_node["eow"] = True
                new_trie.list.append(current_word)
                return True

            # Explore all children
            valid_child = False
            
            # Create list of items once for performance
            items = list(cur.items())
            for char, next_node in items:
                # Skip gray letters (optimization: early pruning)
                if char in gray_set and not yellow.get(char, 0) and char not in green_values:
                    continue

                # Create child node
                new_child = {}
                
                # Update yellow counts only if needed
                if char in yellow and yellow[char] > 0:
                    # Only copy if needed
                    updated_yellow = yellow.copy()
                    updated_yellow[char] -= 1
                else:
                    updated_yellow = yellow

                # Recurse
                if dfs(next_node, current_word + char, updated_yellow, new_child):
                    new_node[char] = new_child
                    valid_child = True

            return valid_child

        dfs(self.root, "", yellow, new_trie.root)
        return new_trie

# More efficient way to compute feedback string
def get_matches(source, target):
    """
    Compute the feedback pattern encoded in 3-ary digits.
    """
    score = 0
    # Precompute target characters for faster lookup
    for i in range(len(target)):
        if source[i] == target[i]:
            score += 2 * (3**i)
        elif source[i] in target:
            score += 3**i
    return score

class Guesser:
    def __init__(self, manual):
        # Load word list once
        with open("wordlist.yaml") as f:
            self._word_list = yaml.load(f, Loader=yaml.FullLoader)
            
        self._manual = manual
        self.console = Console()
        self._tried = []
        
        # Build trie once at initialization
        self._word_trie = Trie()
        for word in self._word_list:
            self._word_trie.insert(word)
        
        # Initialize pattern cache
        self._pattern_cache = {}
        
        # Word frequencies
        self.letter_frequency_abs = Counter()
        self.letter_frequency_pos = {}
        
        # Calculate opening word once
        self._tried_letter = set()
        self.word_list = self._word_list
        self.update_frequencies()
        self.opening = self.frequency_guess_non_word()
        
    def restart_game(self):
        self._tried = []
        self._tried_letter = set(self.opening)
        self.word_list = self._word_list
        self.word_trie = None

    def get_guess(self, result):
        """
        This function must return your guess as a string.
        """
        if self._manual == "manual":
            return self.console.input("Your guess:\n")
        
        # Filter words based on result
        if (self._tried and len(result) == 5):
            self.subset_trie(result)
        
        # Use opening word for first guess
        if not self._tried:
            guess = self.opening
        else:
            guess = self._get_guess()
        
        # Avoid crashes
        while (guess in self._tried
               or len(guess) != 5):
            guess = np.random.choice(
                list(''.join(self._tried)),
                size=5)
            guess = ''.join(guess)

        self._tried.append(guess)
        self.console.print(guess)
        return guess

    def subset_trie(self, result):
        """
        Update the word trie based on feedback string.
        """
        if not self._tried:
            return

        if len(result) != 5:
            print("[!] len(result) > 5. Skipping.")
            return

        guess = self._tried[-1]

        # Initialize feedback containers
        self.yellow = Counter()
        self.gray = set()
        self.green_pos = {}
        self.gray_pos = {}

        # Process feedback from result
        for i, letter in enumerate(guess):
            if result[i] == "+":
                self.gray.add(letter)
                self.gray_pos[i] = letter
            elif result[i] == "-":
                self.yellow.update([letter])
                self.gray_pos[i] = letter
            elif result[i].isalpha():
                self.green_pos[i] = letter

        # Choose parent trie
        parent_trie = self.word_trie if hasattr(self, 'word_trie') and self.word_trie else self._word_trie

        # Generate new filtered trie
        self.word_trie = parent_trie.filter_words(
            gray=self.gray,
            gray_pos=self.gray_pos,
            yellow=self.yellow,
            green_pos=self.green_pos,
            guess=guess,
        )
        self.word_list = self.word_trie.list

    def update_frequencies(self, abs=True, pos=True):
        """
        Update absolute and by-position frequency of letters.
        """
        if not self.word_list:
            return
            
        # Letter Frequencies
        if abs:
            self.letter_frequency_abs = Counter(''.join(self.word_list))

        # Letter Frequencies by Position
        if pos:
            self.letter_frequency_pos = {}

            for i in range(5):
                self.letter_frequency_pos[i] = Counter(word[i] for word in self.word_list)

    def get_shortcut_words(self):
        """
        Synthetically create words joining the remaining letters not yet tried.
        """
        # Calculate letters left only once
        unique_letters = set(self.letter_frequency_abs.keys())
        green_letters = set(self.green_pos.values())
        yellow_letters = set(self.yellow.keys())
        letters_left = unique_letters - green_letters - yellow_letters
        
        if len(letters_left) > 5:
            return []
        
        # Create words to exhaust all letters
        shortcut_word = ''.join(sorted(letters_left, key=lambda x: random()))
        letters_left = list(letters_left)
        if len(shortcut_word) < 5:
            shortcut_word += ''.join(
                np.random.choice(
                    letters_left,
                    size = 5 - len(shortcut_word)
                    ).tolist()
                )
        
        return [shortcut_word]

    def get_max_entropy_word(self, sources, targets):
        """
        Return the word with highest entropy among the provided ones.
        """
        max_entropy = float("-inf")
        best_guess = ""
        tot_counts = len(targets)
        
        # Exit early if no targets
        if tot_counts == 0:
            return sources[0] if sources else "", 0
        
        # Pre-compute log2 values for all possible probabilities
        log2_cache = {1/tot_counts: np.log2(1/tot_counts)}
        
        for w1 in sources:
            # Get all patterns and store the occurrences
            pattern_distribution = Counter()
            
            # Initialize cache for this word if needed
            if w1 not in self._pattern_cache:
                self._pattern_cache[w1] = {}
            cur = self._pattern_cache[w1]
            
            # Calculate pattern distribution
            for w2 in targets:
                if w2 not in cur:
                    cur[w2] = get_matches(w1, w2)
                pattern_distribution[cur[w2]] += 1
            
            # Compute entropy efficiently
            entropy = 0
            for count in pattern_distribution.values():
                prob = count / tot_counts
                # Cache log computations
                if prob not in log2_cache:
                    log2_cache[prob] = np.log2(prob)
                entropy -= prob * log2_cache[prob]

            # Update best guess
            if entropy > max_entropy:
                best_guess = w1
                max_entropy = entropy

        return best_guess, max_entropy

    def frequency_guess_non_word(self):
        """
        Return the combination of top 3 letters by position whose entropy is the highest.
        """
        # Get position-based frequencies
        letters_by_position = []
        for i in range(5):
            i_freqs = self.letter_frequency_pos[i]
            letters_by_position.append([k for k, v in i_freqs.most_common(3)])

        # Generate candidates with no repeated letters
        candidates = []
        
        # Generate candidates using top letters
        for combination in product(*letters_by_position):
            if len(set(combination)) == 5:  # No repeated letters
                candidates.append(''.join(combination))
        
        # If no candidates, combine top letters by frequency and return it
        if not candidates:
            return ''.join(k for k,v in self.letter_frequency_abs.most_common(5))
            
        guess, _ = self.get_max_entropy_word(candidates, self.word_list)

        return guess

    
    def _get_guess(self):
        self.update_frequencies()
        remaining_words = len(self.word_list)
        
        # Early return for empty word list
        if remaining_words == 0:
            return ""
        
        # If only one word left, return it immediately
        if remaining_words == 1:
            return self.word_list[0]
        
        # Use entropy when the list is manageable
        if remaining_words < 50:
            # Generate shortcut words when appropriate
            shortcut_words = []
            if remaining_words > 2 and len(self._tried) < 5:
                shortcut_words = self.get_shortcut_words()
                
            # Find word with maximum entropy
            guess, entropy = self.get_max_entropy_word(
                sources=self.word_list + shortcut_words, 
                targets=self.word_list
            )
            
            # Use entropy-based guess if it's close to optimal
            max_possible_entropy = np.log2(remaining_words)
            if (max_possible_entropy - entropy < 0.5
                or len(self._tried) >= 5):
                return guess
                
        # Fall back to frequency-based guessing
        return self.frequency_guess_non_word()