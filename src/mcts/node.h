/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <memory>
#include <mutex>
#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h"
#include "neural/writer.h"
#include "utils/mutex.h"

namespace lczero {

class Node;
class Node_Iterator {
 public:
  Node_Iterator(Node* node) : node_(node) {}
  Node* operator*() { return node_; }
  Node* operator->() { return node_; }
  bool operator==(Node_Iterator& other) { return node_ == other.node_; }
  bool operator!=(Node_Iterator& other) { return node_ != other.node_; }
  void operator++();

 private:
  Node* node_;
};

class Node {
public:
	// Resets all values (but not links to parents/children/siblings) to zero.
	void ResetStats();

	// Allocates a new node and adds it to front of the children list.
	// Not thread-friendly.
	Node* CreateChild(Move m);

	// Gets parent node.
	Node* GetParent() const { return parent_; }

	// Gets first child
	Node* GetFirstChild() const { return child_; }

	// Returns whether a node has children.
	bool HasChildren() const { return child_ != nullptr; }

	// Returns whether a node has only one child
	bool HasOnlyOneChild() const { return (child_ == nullptr) ? false : (child_->sibling_ == nullptr); }

	// Returns move from the point of new of player BEFORE the position.
	Move GetMove() const { return move_; }

	// Returns move, with optional flip (false == player BEFORE the position).
	Move GetMove(bool flip) const;

	// Returns sum of probabilities for visited children.
	float GetVisitedPolicy() const;
	uint32_t GetN() const { return n_; }
	uint32_t GetNInFlight() const { return n_in_flight_; }
	uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 1; }
	uint32_t GetRealChildrenVisits();
	// Returns n = n_if_flight.
	int GetNStarted() const { return n_ + n_in_flight_; }
	// Returns Q if number of visits is more than 0,
	float GetQ(float default_q) const { return n_ ? q_ : default_q; }
//	float GetQ(float default_q) const { return (n_ || is_certain_) ? q_ : default_q; }
	// Returns U / (Puct * N[parent])
	float GetU() const { return p_ / (1 + n_ + n_in_flight_); }
	// Returns value of Value Head returned from the neural net.
	float GetV() const { return v_; }
	// Returns the avg. of children branches (not expanded grandchildren)
	float GetCB() const { return avg_child_branches_; }
	// Returns value of Move probabilityreturned from the neural net.
	// (but can be changed by adding Dirichlet noise).
	float GetP() const { return p_; }
	// returns branches of this node (number of childs)
	float GetB() const {return b_;}
	// Returns population variance of q.
	float GetSigma2(float default_m) const { return  is_certain_ ? 0 : (n_>1 ? m_/(n_-1):default_m); }
	// Returns whether the node is known to be draw/lose/win.
	bool IsTerminal() const { return is_terminal_; }
	// Returns whether the node is known to have a certain score
	bool IsCertain() const { return is_certain_; }
    uint16_t GetFullDepth() const { return full_depth_; }
    uint16_t GetMaxDepth() const { return max_depth_; }
  // makes node uncertain again (used to make root uncertain when search is initialized
  void UnCertain();
  // Sets node avg of all childrens branches
  void SetCB(float val) { avg_child_branches_ = val; }
  // Sets node own value (from neural net or win/draw/lose adjudication).
  void SetV(float val) { v_ = val; }
  // Sets move probability.
  void SetP(float val) { p_ = val; }
  // Sets Q
  void SetQ(float val) { q_ = val; }
  // Sets branches (number of childs)
  void SetB(float val) { b_ = val; }
  // Sets n_ for terminal nodes that are 
  // found when creating children in 
  // expand node
  void SetN(uint32_t const n) { n_ = n; }
  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result);
  // Makes the node certain and sets it's score
  void MakeCertain(float q);

  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate();

  // Updates the node with newly computed value v.
  // Updates:
  // * N (+=1)
  // * N-in-flight (-=1)
  // * W (+= v)  obsolete
  // * Q (+= q + (v - q) (n_+1))
  // * M Sum of Squares of Differences
  // kBackpropagate (not used currently) and Autoextend modes are 
  // currently passed as parameters
  // will either be removed if changes become permanent, or replaced
  // by a weight parameter.
  void FinalizeScoreUpdate(float v, int kAutoextend);

  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);

  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);

  V3TrainingData GetV3TrainingData(GameResult result,
                                   const PositionHistory& history) const;

  class NodeRange {
   public:
    Node_Iterator begin() { return Node_Iterator(node_); }
    Node_Iterator end() { return Node_Iterator(nullptr); }

   private:
    NodeRange(Node* node) : node_(node) {}
    Node* node_;
    friend class Node;
  };

  // Returns range for iterating over children.
  NodeRange Children() const { return child_; }

  // Debug information about the node.
  std::string DebugString() const;

  class Pool;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Q value fetched from neural network.
  float v_;
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. Terminal nodes (which lead to checkmate or draw) may be visited
  // several times, those are counted several times. q = w / n
  float q_;
  // Sum of values of all visited nodes in a subtree. Used to compute an
  // average. No longer needed in may version but kept for debug reasons
  float w_;
  // Probabality that this move will be made. From policy head of the neural
  // network.
  float p_;
  // Sum of Squares of Differences from current mean
  float m_;
  // branch data for tree shaping
  float b_;
  float avg_child_branches_;
  // How many completed visits this node had.
  uint32_t n_;
  // (aka virtual loss). How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  uint16_t n_in_flight_;

  // Maximum depth any subnodes of this node were looked at.
  uint16_t max_depth_;
  // Complete depth all subnodes of this node were fully searched.
  uint16_t full_depth_;
  // Does this node end game (with a winning of either sides or draw).
  bool is_terminal_;
  // Is this nodes q certain (this is true if the node's children are all certain)
  // all terminal nodes are certain
  bool is_certain_;
  // Pointer to a parent node. nullptr for the root.
  Node* parent_;
  // Pointer to a first child. nullptr for leave node.
  Node* child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  Node* sibling_;

  // TODO(mooskagh) Unfriend both NodeTree and Node::Pool.
  friend class NodeTree;
  friend class Node_Iterator;
};

inline void Node_Iterator::operator++() { node_ = node_->sibling_; }

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  // Adds a move to current_head_;
  void MakeMove(Move move);
  // Sets the position in a tree, trying to reuse the tree.
  void ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_; }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  Node* current_head_ = nullptr;
  Node* gamebegin_node_ = nullptr;
  PositionHistory history_;
};
}  // namespace lczero