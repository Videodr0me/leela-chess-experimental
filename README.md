# leela-chess-experimental
based on Leela Chess Zero https://github.com/LeelaChessZero

In order to familiarize myself with the code and get a feel for MCTS, i tried a number of search ideas in lc0. If you find something interesting here feel free to open an issue and discuss. This is a work in progress and purely experimental - new ideas will be added from time to time. This serves as documentation of both the good as well as the bad tries - so do not expect huge gains - but some ideas yield a measurable elo gain.  

Disclaimer: All changes are completely zero, completely game agnostic and need no parameters (except one). 
## Sanity Tests

### FPU or not to FPU
Deep Mind is somewhat vague in their papers on if and how they use FPU. At one point they claim to not use FPU at all, and at another they preinitialize q with 0. I tested all common FPU approaches with these results for chess. Baseline is -parent q:

FPU type | match result
------- | -------------------
q = 0| +125 -443 =432 Win: 34.10% Elo: -114.45 LOS:  0.00%
q = 1.1 (No FPU)| +6 -905 =89 Win:  5.05% Elo: -509.68 LOS:  0.00%
q = -parent_v | +262 -307 =431 Win: 47.75% Elo: -15.65 LOS:  2.96%

all tests with FPU-Reductions disabled (=0.0).
One can safely conclude that FPU with -parent_q is strongest. And I strongly suspect this is what Deep Mind used, at least in Alpha Zero. Maybe we will know more if the full paper is published.


## Search Modifications

### Single-legal-move extension

For a decription and an example position of what this does see:
https://github.com/Videodr0me/leela-chess-experimental/wiki#single-legal-move-positions---auto-extend1

10.000 Game self-play result:
```
tournamentstatus final P1: +2974 -2925 =4101 Win: 50.24% Elo:  1.70 LOS: 73.82% P1-W: +1705 -1267 =2028 P1-B: +1269 -1658 =2073
```

Result is within expectations, as this minor change takes probably 100.000 games to properly assess.

### Certainty propagation 

This is also known as MCTS-Solver or Proof-Number-Search in literature. For a description with example positions see: https://github.com/Videodr0me/leela-chess-experimental/wiki#certainty-propagation---certainty-prop1

```
tournamentstatus final P1: +2937 -2849 =4214 Win: 50.44% Elo:  3.06 LOS: 87.63% P1-W: +1718 -1140 =2142 P1-B: +1219 -1709 =2072
```

Result is also within expectation, but this has some nice properties. Certain winning moves at root can be played regardless of visit counts, which is beneficial in time pressure situations as MCTS is slow to revise initial estimates. Also if one day Tablebases get added, certainty propagation is useful for propagating the TB probe results throughout the tree. 

### Moving-Average-Q
This is my flavor of Gudmundsson and Björnsson 2011. For a description see:

https://github.com/Videodr0me/leela-chess-experimental/wiki#power-decay-averaging-tree-search


Match | power gamma=0.75 | linear beta=0.49 | limited beta
------- | ------------------- | ------------ | ----------- 
10000 Games / 100 Visits|  Elo: 24.60 LOS: 100.00% | Elo: 45.04 LOS: 100.00% | pending
1000 Games / 800 Visits |     Elo: -4.52 LOS: 25.04% | Elo:  4.17 LOS: 72.38% | pending
500 Games / 10000 Visits | not planned | Elo: -11.12 LOS:  4.94% | pending

### UCB1 tuned and other variance based approaches

pending

## Validation run 1

10.000 game validation run with all three options enabled (uncertainty-prop=1, auto-extend=1, backpropagate-gamma=0.75)
```
tournamentstatus final P1: +3220 -2343 =4437 Win: 54.39% Elo: 30.55 LOS: 100.00% P1-W: +1721 -1184 =2095 P1-B: +1499 -1159 =2342
```
Result confirms above single option results and strength contributions seem additive at low visit searches per move. Higher visit match games pending...
