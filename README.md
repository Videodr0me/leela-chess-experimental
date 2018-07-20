# leela-chess-experimental
based on Leela Chess Zero https://github.com/LeelaChessZero

Update 20/07/2018: V3.1 released, added two-fold draw scoring, dynamic policy compression, minor bugfix
https://github.com/Videodr0me/leela-chess-experimental/releases/tag/v3.1


Update 01/07/2018: V3.0 released, includes empirical variance, and policy-compression-decay. 

Update 24/06/2018: First results for variance based approach, many results for tree balancing.

Update 19/06/2018: V2.1, new source, new executable, changes to tree balancing, easy-early-visits

Update 17/06/2018: New source, new executable, new parameters and new test results.

I tried a number of MCTS search ideas in lc0. If you find something interesting here feel free to open an issue and discuss. This is a work in progress and purely experimental - new ideas will be added from time to time. This serves as documentation of both the good as well as the bad tries - so do not expect huge gains - but some ideas yield a measurable elo gain.  

Disclaimer: All changes are completely zero and game agnostic.

## Search Modifications
### Empirical variance based approaches (similar to UCB1-tuned and others) Work in Progress

Empirical variance of q is calculated for each node and used for node selection. The standard upper confidence bound schemes (and leelas/A0 flavour of it)  estimates the implied variance based on sample size (parent in relation to child). These experiments use an estimate based on actual backpropagated q values. Variances are calculated with a numerically robust "online" algorithm and can be displayed by --verbose-movestats. The following is a first (crude) version that shows this can work (1000 games, 2000 visits per move). 

```
P1: +159 -139 =702 Win: 51.00% Elo:  6.95 LOS: 87.67% P1-W: +109 -48 =343 P1-B: +50 -91 =359
````
The current version works by adding half normal random noise with mean 0 and the empirical estimate of each nodes variance. This is scaled by a variance-scaling-factore (0.01 in the above match). For unexpanded nodes the parents variance is used. This approach servers as a first proof of concept and can be refined in many ways. It assumes that there an amount of unexplained variance outside of the policy weighted implied UCB. More traditional (non MC) approaches that combine the emprical variance with the implied variance (like UCB1 tuned) did not work very well, often overestimating or underestimating the ucb - probably due to overly simplistic distributional assumptions. One advantage of this simple solution is that it brings the MC back in MCTS, but most likely it can be improved by a more thorough statistical analysis.  

Source and executable soon.

### Tree Balancing - Work in Progress
The upper confidence bound used in LC0's UCT flavor (and that of A0) assumes that the confidence bound of a child node is not affected by the local branching factor. However, in some games like Draughts or Chess the number of legal moves (branches) can vary greatly even in the same part of the search tree. This modification is based on the idea that we can use the number of individual branches in relation to the average number of branches to adjust the upper bound when selecting child nodes for expansion. Parameters are:

--tree-balance
--tree-scale-left
--tree-scale-right

Parameters | Match | match result| Elo
---------- | ------| --------|----
tb=1.5 tsl=1.5 tsr=(1)|V=800 G=1000  |P1: +212 -122 =666 Win: 54.50% |Elo: 31.35 LOS: 100.00% 
tb=1.4 tsl=2.0 tsr = 0.1|V=2000 G=647 | P1: +90 -80 =477 Win: 50.77% |Elo:  5.37 LOS: 77.84% 
tb=1.1 tsl=2.0 tsr =0.3| V=2000 G=1000| P1: +140 -120 =740 Win: 51.00% |Elo:  6.95 LOS: 89.26% 
tb=1.07 tsl=2.0 tsr=0.2| V=2000 G=1000 |P1: +136 -111 =753 Win: 51.25% |Elo:  8.69 LOS: 94.42% 
tb=1.3 tsl=1.5 tsr=0.01| V=10000 G=500 |P1: +49 -42 =409 Win: 50.70% |Elo:  4.86 LOS: 76.85% 

(1) This version did not use the --tree-scale-right parameter, but setting it to 0.01 will yield identical results

Results seem stable, rating compression on higher visit games is expected due to higher draw rate. A next step would be to  "CLOP" the parameters (preferably at high(er) visit counts). I am still not quite satisfied with the current solution (parameters are much more extreme than my back-of-the-evenlope math suggests) and am working on a variant that focusses more on using the branching information for first node expansions.

More tests to follow

### Certainty propagation & Single-legal-move extension

A variant of MCTS-Solver (Winands et. al). For a decription, example positions and self-play elo see:

https://github.com/Videodr0me/leela-chess-experimental/wiki/MCTS-Solver---Certainty-Propagation-and-Autoextending

```
P1: +181 -152 =667 Win: 51.45% Elo: 10.08 LOS: 94.40% P1-W: +102 -69 =328 P1-B: +79 -83 =339
```
Besides the small gain of elo, this has some additonal nice properties. Leela now finds shallow mates faster and certain winning moves at root can be played regardless of visit counts, which is beneficial in time pressure situations (typically MCTS is slow to revise initial estimates).

Update: 
Improvement (v2.1) - if current best move is a certain loss change best move to a non loosing move, even if visits are lower.

### Compress low policy move probabilites
Instead of changing softmax temperature this scheme encourages exploration of low policy priors by compressing low probabilites more than high probabilites in relation to search depths. This was initially devised to do well at tactics, which usually looses some elo in self-play. But to my suprise when i finally came around to test this at suffiently high visit searches it seems to gain in self-play elo as well. Thanks to EXA for making me aware of this:

```
P1: +65 -39 =521 Win: 52.08% Elo: 14.46 LOS: 99.46% P1-W: +41 -13 =258 P1-B: +24 -26 =263
``` 

This is with --policy-compression=0.06 vs. standard leela at 10000 visits per move.

This might also be useful for long analysis as it restores MCTS convergence properties (under some circumstances leela would never find moves no matter how many nodes visited.). 

### Easy Early Visits:
Tweaks the formula slightly to encourage early visits. The confidence bound is asymptotically unchanged but second and to a lesser degree third visits are more probable.
```
--easy-early-visits=0 (disabled)
--easy-early-visits=1 (enabled)
```
Might help ever so slightly tactically - this is untested but might work well in conjunction with policy compression. Self-play might suffer, but untested against non-leela opponents. 

Update: This is now float parameter with 0.0 turning this feature off and 1.0 corresponding to old enabled behavior. Now values between 0.0 and 1.0 are also possible.

### Q-Moving-Average
Tested some variants of Gudmundsson and Björnsson and Feldman and Domshlak. For a description see:

https://github.com/Videodr0me/leela-chess-experimental/wiki/Backpropagation:-Q-Moving-Average

Large elo gains at low visit counts, but doesn't currently scale with larger visit searches. Interesting but needs further investigation.

### Do not trust initial visits q (fully)
Similar to FPU this assumes that when selecting nodes for expansion that first backpropagated NN eval is still unreliable and gets averaged with parent-q for PUCT evaluation. Details of multiple flavours here:

https://github.com/Videodr0me/leela-chess-experimental/wiki/Selection:-Don't-trust-initial-visits

Inconclusive results or elo losses. Could not make this work.




## Validation Runs

### Run for Leela MCTS Experimental v2
coming soon

### Run for Leela MCTS Experimental v1

10.000 game (100 visits per move) validation run with these options enabled (uncertainty-prop=1, auto-extend=1, backpropagate-gamma=0.75)
```
tournamentstatus final P1: +3220 -2343 =4437 Win: 54.39% Elo: 30.55 LOS: 100.00% P1-W: +1721 -1184 =2095 P1-B: +1499 -1159 =2342
```
Unfortunately backpropagate-gamma does not scale for larger visit searches. Certainty-propagation has been improved since the above match. The next validation run with all modifications that gain elo will be done as soon as testing of new schemes is complete.


## Miscellaneous
A number of "sanity tests" of FPU variants, Cpuct and various minor MCTS tweaks can be found here. These are mainly done to corroborate that current leela is working as expected and to restest some minor implementation details in chess:
https://github.com/Videodr0me/leela-chess-experimental/wiki/Sanity-Tests

### Notes
Test parameters were old default cpuct=1.2, fpu-reduction=0.0, and NN 5d46d9c438a6901e7cd8ebfde15ec00117119cabfcd528d4ce5f568243ded5ee

For test positions threads=1 and batchsize=1 were used for reproducability.
