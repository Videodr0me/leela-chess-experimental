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

#include "selfplay/loop.h"
#include "selfplay/tournament.h"
#include <sstream>
#include <iostream>
#include <iomanip>


namespace lczero {

namespace {
const char* kInteractive = "Run in interactive mode with uci-like interface";
}  // namespace

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  options_.Add<BoolOption>(kInteractive, "interactive") = false;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  if (options_.GetOptionsDict().Get<bool>(kInteractive)) {
    UciLoop::RunLoop();
  } else {
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  SendResponse("id name The Lc0 chess engine.");
  SendResponse("id author The LCZero Authors.");
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  options_.SendAllOptions();
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  SendResponse(res);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
	int winp1 = info.results[0][0] + info.results[0][1];
	int loosep1 = info.results[2][0] + info.results[2][1]; 
	int draws = info.results[1][0] + info.results[1][1];
    float perct=-1, elo=99999;
	float los = 99999;
    if ((winp1+loosep1+draws)>0) perct = (((float)draws) / 2 + winp1) / (winp1 + loosep1 + draws);
	if ((perct < 1) && (perct > 0)) elo = -400 * log(1 / perct - 1) / log(10);
	if ((winp1 + loosep1)>0) los = .5 + .5 * std::erf((winp1 - loosep1) / std::sqrt(2.0*(winp1 + loosep1)));

	std::string res = "tournamentstatus";
    if (info.finished) res += " final";
    res += " P1: +" + std::to_string(winp1) + " -" + std::to_string(loosep1) + " =" + std::to_string(draws);

    if (perct > 0) {
 	  std::ostringstream oss;
	  oss << std::fixed<<std::setw(5) << std::setprecision(2) << (perct * 100) <<"%";
	  res += " Win: " + oss.str();
    }
    if (elo < 99998) {
	  std::ostringstream oss;
	  oss << std::fixed<< std::setw(5)<<std::setprecision(2) << (elo);
	  res += " Elo: " + oss.str();
    }
	if (los < 99998) {
		std::ostringstream oss;
		oss << std::fixed << std::setw(5) << std::setprecision(2) << (los * 100) << "%";
		res += " LOS: " + oss.str();
	}
    res += " P1-W: +" + std::to_string(info.results[0][0]) + " -" + std::to_string(info.results[2][0]) + " =" + std::to_string(info.results[1][0]);
    //Might be redundant to also list P1-B:
    res += " P1-B: +" + std::to_string(info.results[0][1]) + " -" + std::to_string(info.results[2][1]) + " =" + std::to_string(info.results[1][1]);
    SendResponse(res);
}

}  // namespace lczero