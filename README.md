# RL project: Gumbel AlphaZero chess with a metareasoning controller

DRE 7060 Reinforcement Learning for Business (BI, 2026/2027).

A chess agent trained with a supervised warm start and Gumbel AlphaZero self-play, and a
controller that decides per move whether to play, search deeper, consult Stockfish, or
consult a judge, trading playing strength against computation cost.

All code, documentation and usage instructions are in [`code/`](code/README.md).
Data (`data/`), engines (`engines/`) and training runs (`runs/`) are not tracked; they are
recreated by the scripts in `code/`.
