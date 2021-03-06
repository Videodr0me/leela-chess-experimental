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

#include <iostream>
#include "analyzer/analyzer.h"
#include "engine.h"
#include "selfplay/loop.h"
#include "utils/commandline.h"

int main(int argc, const char** argv) {
  std::cerr << "       _" << std::endl;
  std::cerr << "|   _ | |" << std::endl;
  std::cerr << "|_ |_ |_| MCTS EXPERIMENTAL " << __DATE__ << std::endl;
  using namespace lczero;
  CommandLine::Init(argc, argv);
  CommandLine::RegisterMode("uci", "(default) Act as UCI engine");
  CommandLine::RegisterMode("selfplay", "Play games with itself");
  CommandLine::RegisterMode("debug", "Generate debug data for a position");

  if (CommandLine::ConsumeCommand("selfplay")) {
    // Selfplay mode.
    SelfPlayLoop loop;
    loop.RunLoop();
  } else if (CommandLine::ConsumeCommand("debug")) {
    // Runs analyzer mode.
    Analyzer analyzer;
    analyzer.Run();
  } else {
    // Consuming optional "uci" mode.
    CommandLine::ConsumeCommand("uci");
    // Ordinary UCI engine.
    EngineLoop loop;
    loop.RunLoop();
  }
}
