# Solving rubik's cubes with RL

## Repo overview: how to use

This repo was made for the 2023 kaggle competition on Polytope permutation: https://www.kaggle.com/competitions/santa-2023/overview, the notations for each puzzle moves & conditions are similar to the competition.

## Task Definition: Arbitrary Permutation puzzles
We generalize the problem of solving the particular puzzles in the polytope permutation competition (cube, wreath, globe) to solving *arbitrary* permutation puzzles. This is a very well defined Markov Decision Process with perfect observability, finite horizon, finite state (although practically infinite) space and finite action space.

## Environment Generation
* Permutation as swaps 
* Greedy reduction + link to kaggle
* Generate positions


## Baseline: Vanilla Deep Q-learning
* Quick definitions, pointers to papers
* Configs

## Improvements from baseline

### Double Learning

### N-step return

### Noisy Network 

