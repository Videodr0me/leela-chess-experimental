# leela-chess-experimental
based on Leela Chess Zero https://github.com/LeelaChessZero

Update on Cetainty Propagation 11/06/2018

I tried a number of MCTS search ideas in lc0. If you find something interesting here feel free to open an issue and discuss. This is a work in progress and purely experimental - new ideas will be added from time to time. This serves as documentation of both the good as well as the bad tries - so do not expect huge gains - but some ideas yield a measurable elo gain.  

Disclaimer: All changes are completely zero, completely game agnostic and need no parameters (except one). 

## Search Modifications

### Certainty propagation & Single-legal-move extension

A variant of MCTS-Solver (Winands et. al). For a decription, example positions and self-play elo see:

https://github.com/Videodr0me/leela-chess-experimental/wiki/MCTS-Solver---Certainty-Propagation-and-Autoextending

```
P1: +181 -152 =667 Win: 51.45% Elo: 10.08 LOS: 94.40% P1-W: +102 -69 =328 P1-B: +79 -83 =339
```
Besides the small gain of elo, this has some additonal nice properties. Leela now finds shallow mates faster and certain winning moves at root can be played regardless of visit counts, which is beneficial in time pressure situations (typically MCTS is slow to revise initial estimates).

### Q-Moving-Average
Tested some variants of Gudmundsson and Björnsson and Feldman and Domshlak. For a description see:

https://github.com/Videodr0me/leela-chess-experimental/wiki/Backpropagation:-Q-Moving-Average

Large elo gains at low visit counts, but doesn't currently scale with larger visit searches. Interesting but needs further investigation.

## Node Selection for Expansion: 

### Tree-balancing
Soon.


### Compress low policy move probabilites 
Instead of changing softmax temperature this scheme encourages exploration of low policy priors by compressing low probabilites more than high probabilites. See:

Very good on tactics tests, but looses some self-play elo. Not tested against non-leela opponents.

### Do not trust initial visits (fully)
Similar to FPU this assumes that when selecting nodes for expansion that first backpropagated NN eval is still unreliable and gets averaged with parent-q for PUCT evaluation. Details of multiple flavours here:

https://github.com/Videodr0me/leela-chess-experimental/wiki/Selection:-Don't-trust-initial-visits

Inconclusive results or elo losses. Could not make this work.


### UCB1 tuned and other variance based approaches

Variance of q is calculated for each node. And used for node selection. Work in progress: variances are calculated with a numerically robust "online" algorithm. Use --verbose-movestats to display variances for each node. These stats are very interesting, next is to use this info in a theoretically sound way in the PUCT formula.

## Validation Runs

10.000 game (100 visits per move) validation run with these options enabled (uncertainty-prop=1, auto-extend=1, backpropagate-gamma=0.75)
```
tournamentstatus final P1: +3220 -2343 =4437 Win: 54.39% Elo: 30.55 LOS: 100.00% P1-W: +1721 -1184 =2095 P1-B: +1499 -1159 =2342
```
Unfortunately backpropagate-gamma does not scale for larger visit searches. Certainty-propagation has been improved since the above match. The next validation run with all modifications that gain elo will be done as soon as testing of new schemes is complete.


## Miscellaneous
A number of "sanity tests" of FPU variants, Cpuct and various minor MCTS tweaks can be found here. These are mainly done to corroborate that current leela is working as expected and to restest some minor implementation details in chess:
https://github.com/Videodr0me/leela-chess-experimental/wiki/Sanity-Tests
