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
| `model.py` | ResNet with policy + WDL value head; device auto-detection (xpu → cuda → mps → cpu) |
| `mcts/gumbel.py` | Gumbel AlphaZero search (sequential halving with the mctx visit schedule, completed-Q improved policy, min-max Q normalisation); generator-based leaf batching across games |
| `training/selfplay.py` | Batched self-play with playout-cap randomisation, resignation on the search value, and length adjudication |
| `training/resign.py` | Resignation threshold calibrated on no-resign games (false positives <= 5%, AlphaGo Zero) |
| `training/logger.py` | JSONL + TensorBoard logging (per step, per iteration, per evaluation) |
| `training/replay_buffer.py` | Compact ring buffer (uint8 planes, sparse policy targets) |
| `training/losses.py` | Policy CE + WDL CE training step |
| `download.py` | Downloads Stockfish and Lichess Elite PGNs |
| `warmstart.py` | Supervised pretraining on Elite games |
| `train_selfplay.py` | Main loop: self-play → buffer → train → checkpoint → periodic Elo evaluation (exactly resumable) |
| `evaluate.py` | Parallel, colour-paired matches vs. Stockfish (UCI_Elo / Skill Level) or checkpoints, Elo with 95% CI |
| `benchmark.py` | CPU vs. GPU (Intel XPU / Apple MPS) throughput (inference, self-play, training) |
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
```
Training logs are written to `runs/<run_name>/`:
- `steps.jsonl`: every gradient step (loss, policy/value loss and accuracy, gradient norm, lr)
- `log.jsonl`: every iteration (games, terminations, search-target entropy and KL to the prior,
  value error, resignation threshold and false-positive rate, buffer reuse, timings)
- `eval.jsonl`: every `TrainEvalConfig.every` iterations: Elo vs. Stockfish (opponent set to the previous
  estimate), match vs. the previous snapshot, and seconds per move with policy only and with search
- `tensorboard/`: the same scalars, `tensorboard --logdir runs`

`latest.pt` and `buffer.npz` are written together every `checkpoint_every` iterations (with RNG states),
so a rerun resumes exactly from the last checkpoint.

## Python environments
Runs on Windows, macOS and Linux. `download.py stockfish` fetches the Stockfish build for the current
OS / CPU, and `get_device` smoke-tests the GPU and falls back to CPU if it fails.

- **Windows (Intel Arc 140V):** use `../.venv-xpu/Scripts/python.exe` (PyTorch XPU build, ~2.6x faster
  self-play and ~6x faster training than CPU). The system Python (CPU torch) also works.
- **macOS (Apple Silicon):** create `../.venv` with Python 3.14 or older (PyTorch has no wheels for newer
  versions yet), e.g. `python3.14 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt`.
  The standard PyTorch build uses the MPS GPU automatically (~3.7x faster self-play and ~9x faster
  training than CPU).

## References
- Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022.
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*, Science 2018.
- Wu, *Accelerating self-play learning in Go* (KataGo), 2019 — playout cap randomisation.
