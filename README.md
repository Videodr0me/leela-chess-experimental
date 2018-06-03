# leela-chess-experimental
based on Leela Chess Zero https://github.com/LeelaChessZero

In order to familiarize myself with the code and get a feel for MCTS, i tried a number of search ideas in lc0. If you find something interesting here feel free to open an issue and discuss. This is a work in progress and purely experimental - new ideas will be added from time to time. This servers more as documentation of the good as well as the bad tries - so do not expect huge gains but some ideas yield an measurable elo gain.  

Disclaimer: All changes are completely zero, completely game agnostic and need no parameters (except one). 

## Search Modifications

### Single-legal-move positions
Nodes (or chains of nodes) that have only one-legal-move are game theoretically completely linked. For example: Node A -> only one legal move -> Node B -> only one legal move -> Node C -> Many legal moves. Here A, B and C while different nodes are actually one game state, as from A you will inevitably land in C. They should have identical evaluations. Currently MCTS first vists A, propagates the score, then maybe visits B propagates the score and only on the third visit reaches C - while actually it could have directly  visited C and backed up that score to B and A. Not only would this save visits, but it is also probable that a deeper NN eval is more accurate then a shallow one (Beware: this is not always the case!). I called this auto-extending, because if only one legal move, we really want to first eval C and then just back up the score.

In chess these situations most often occur after a check, where the opponents King has only one move. In other games like Othello or Checkers there are even more prevalent. One example Position with Leelas normal MCTS (all tests with NN 5d46d9c438a6901e7cd8ebfde15ec00117119cabfcd528d4ce5f568243ded5ee):

```
position fen 5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - 0 1
go nodes 30000
.
.
info depth 2 seldepth 28 time 5179 nodes 19476 score cp -3 hashfull 40 nps 3760 pv c6d6 h3c8 f8f7 c8b7 f7g8 d1g1 f2f7 b7a8 f7f8 a8b7 f8f7 b7a8 f7f8 a8b7 f8f7
info depth 2 seldepth 28 time 5192 nodes 20254 score cp 9003 hashfull 40 nps 3901 pv c6c4 d6c4 b5c4
```

It takes normal MCTS 20254 nodes to find the correct move Qc4. 

With auto-extending (--auto-extend=1) it takes only 10340 nodes and time went down from 5192 to 2923ms:

```
info depth 2 seldepth 27 time 2919 nodes 10215 score cp 1 hashfull 23 nps 3499 pv c6d6 h3c8 f8f7 c8b7 f7g8 d1g1 f2f7 b7a8 f7f8 a8b7 f8f7 b7a8 f7f8 a8b7 f8f7
info depth 2 seldepth 27 time 2923 nodes 10340 score cp 7304 hashfull 23 nps 3537 pv c6c4 d6c4 b5c4
```
Well, one example does of not prove much, so i played a 10.000 game self-play match. I did not expect this change to be measurable, because even in chess such only-move cases are rare and even if they occur it was not clear if they really matter. The result was within those expectations - and probably 100.000 games are necessary to finally get out of the noise levels.

```
lc0-cudnn selfplay --parallelism=8 --backend=multiplexing "--backend-opts=cudnn(threads=2)" --games=10000 --visits=100 --temperature=1 --tempdecay-moves=10 player1: --certainty-prop=0 --auto-extend=1 --backpropagate-mode=0 --optimal-select=0 player2: --auto-extend=0 --certainty-prop=0 --optimal-select=0 --backpropagate-mode=0

tournamentstatus final P1: +2974 -2925 =4101 Win: 50.24% Elo:  1.70 LOS: 73.82% P1-W: +1705 -1267 =2028 P1-B: +1269 -1658 =2073
```

### Certainty propagation 
This is known in literature as MCTS-Solver or Proof-Number-Search. If we reach terminal (which are "certain") nodes, we can propagate the "certain" results from these nodes more efficiently compared to standard leela MCTS where they are treated the same as nodes evaluate by the NN (and are "uncertain). For example if on the move one side can play a move thats leads to a terminal win node, we can make the parent node a certain. Further, nodes whose children are all certain, become certain themselves with the max q among the certain childs. With this even MCTS can solve some shallow mates, and steer exploring the tree more efficently as some branches are now known to no longer need exploration. Another nice property is that if we have a certain win at root we can play it immediatly, regardless of the visits that move received. 

One example position 


```
lc0-cudnn selfplay --parallelism=8 --backend=multiplexing "--backend-opts=cudnn(threads=2)" --games=10000 --visits=100 --temperature=1 --tempdecay-moves=10 player1: --certainty-prop=1 --auto-extend=0 --backpropagate-mode=0 --optimal-select=0 player2: --auto-extend=0 --certainty-prop=0 --optimal-select=0 --backpropagate-mode=0
tournamentstatus final P1: +2937 -2849 =4214 Win: 50.44% Elo:  3.06 LOS: 87.63% P1-W: +1718 -1140 =2142 P1-B: +1219 -1709 =2072
```
