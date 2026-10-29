# Gumbel AlphaZero chess + metareasoning controller

DRE 7060 course project. Phase 1 (this code): a warm-started, self-play chess agent
using Gumbel AlphaZero search, plus a rating harness. Phase 2 (todo): the controller
that chooses between *play*, *search deeper*, *consult Stockfish*, and *consult judge*.

All parameters and hyperparameters are in `config.py` (dataclasses grouped per component).

## Layout
| File | Purpose |
|---|---|
| `config.py` | All parameters (paths, model, MCTS, self-play, training, warm start, eval) |
| `chess_env/encoding.py` | 20-plane board encoding, 4168-move policy index (side-to-move perspective) |
| `model.py` | ResNet with policy + WDL value head; device auto-detection (xpu → cuda → cpu) |
| `mcts/gumbel.py` | Gumbel AlphaZero search (sequential halving at root, completed-Q improved policy); generator-based leaf batching across games |
| `training/selfplay.py` | Batched self-play with playout-cap randomisation, resignation, and length adjudication |
| `training/replay_buffer.py` | Compact ring buffer (uint8 planes, sparse policy targets) |
| `training/losses.py` | Policy CE + WDL CE training step |
| `download.py` | Downloads Stockfish and Lichess Elite PGNs |
| `warmstart.py` | Supervised pretraining on Elite games |
| `train_selfplay.py` | Main loop: self-play → buffer → train → checkpoint (resumable) |
| `evaluate.py` | Matches vs. Stockfish (UCI_Elo / Skill Level) or checkpoints, Elo with 95% CI |
| `play.py`, `play_ui.html` | Play in the browser vs. a checkpoint (search or raw policy) or Stockfish, point-and-click with chess clock |
| `benchmark.py` | CPU vs. Intel XPU throughput (inference, self-play, training) |
| `controller/features.py` | Controller state: entropy, visit/Q gaps, WDL, value change, material, ... |
| `controller/reward.py` | Offline reward: transform (cp / cp_tanh / expected_score), costs, fault model (none / uniform / uncertainty) |
| `controller/dataset.py` | Counterfactual dataset: oracle eval of base, deeper, expert (Stockfish UCI_Elo 2200) and random-fault moves |
| `controller/bandit.py` | Gain-predicting MLP, lambda-sweep policy, baselines (always-X, thresholds, oracle), Pareto plot |
| `tests/` | Encoding round-trip, search sanity (incl. mate-in-one), self-play smoke test |

## Workflow (run from `code/`)
```
pip install -r requirements.txt
python -m pytest tests
python download.py stockfish
python download.py elite          # months set in DownloadConfig.elite_months
python warmstart.py build
python warmstart.py train         # -> runs/main/warmstart.pt
python train_selfplay.py          # runs until stopped; Ctrl+C and rerun to resume
python evaluate.py stockfish ../runs/main/latest.pt
python evaluate.py match ../runs/main/checkpoints/iter_00100.pt ../runs/main/checkpoints/iter_00010.pt
python play.py                    # play against an agent at http://127.0.0.1:5000
```
Controller (after self-play is running; works with any checkpoint):
```
python -m controller.dataset ../runs/main/latest.pt   # -> runs/main/controller_data.npz
python -m controller.bandit                           # results JSON + Pareto plot
```
The dataset stores raw oracle centipawns and a uniform draw per position, so reward transform,
costs and fault mode (`ControllerConfig`) can be changed and `controller.bandit` rerun without
regenerating data.

Training logs are written to `runs/<run_name>/log.jsonl`.

## Python environments
Use `../.venv-xpu/Scripts/python.exe` (PyTorch XPU build for the Intel Arc 140V, ~2.6x faster self-play
and ~6x faster training than CPU). `get_device` smoke-tests the GPU and falls back to CPU if it fails;
the system Python (CPU torch) also works.

## References
- Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022.
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*, Science 2018.
- Wu, *Accelerating self-play learning in Go* (KataGo), 2019 — playout cap randomisation.
