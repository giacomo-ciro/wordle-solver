# 📚 Wordle Solver
A program to find the optimal next word to guess in the popular online game [Wordle](https://www.nytimes.com/games/wordle/index.html).

To play the game manually:
```
python game.py --p
```

To run multiple games automatically and track performance (`--p` is the verbosity flag):

```
python game.py --p --r 100
```

To change the word lists used, please modify the path in both the `wordle.py` and `guesser.py` files. Available word lists are:
- `train_wordlist.yaml`  <- a set of 4,200 english words
- `dev_wordlist.yaml`    <- a subset of 500 words, used for development
- `r_wordlist.yaml`      <- another random subset of 500 words (more can be generated using the `multiple_runs.py` script)
- `nonsense_wordlist.yaml`  <- a set of 500 random concatenation of 5 letters


## Methodology
At each round, I measure the Shannon Entropy of the remaining words and use the word with the highest one as the optimal guess. Entropy is measured on the probability distribution of the feedback. In particular, for each word left, the feedback is computed against all remaining other words and the counts are used to estimate a probability distribution.

## Results
On a set of 4,200 possible words, I achieved an accuracy of 100% over 50,000 total trials, with 3.7 average number of guesses to guess the word. When restricted to 500 possible words, this average number of guesses dropped down to 2.91.

| Version     | Results                 | Notes                                                                     | 
| :---:     | :--:                      | :--:                                                                      |
| V4        | 100% - 3.7734 - 593.18s   | 50,000 independent runs on train set fo 4,200 words, with printing enabled| 
| FINAL     | 100% - 2.91 - 0.8s        | On 100 random sets of 500 words from train set                            |

\* Results are formatted as (Accuracy, Average Number of Guesses, Running Time) 
