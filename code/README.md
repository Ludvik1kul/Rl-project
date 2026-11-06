# Gumbel AlphaZero chess + metareasoning controller

DRE 7060 course project.

- **Phase 1 (this code):** a chess agent that is first trained by supervised learning on strong
  human games, with Stockfish evaluations as value targets, and then improved by self-play with
  Gumbel AlphaZero search (Danihelka et al., 2022). It includes a rating harness against Stockfish.
- **Phase 2 (planned, to be redesigned):** a controller that decides per move whether extra
  computation or outside help is worth its cost.

Every parameter lives in `config.py` (one dataclass per component). Scripts read `Config()`; tests and
test runs change fields on a copy.

## How it works

1. **Data.** `download.py` fetches Stockfish and Lichess Elite PGNs.
2. **Warm start** (`warmstart.py`).
   - **Positions:** 10 random positions from each game where both players are rated 2000 or more.
   - **Value target:** Stockfish 19 (50k nodes per position, 7 processes in parallel) gives its own
     win/draw/loss estimate, stored as P(win) − P(loss) for the side to move, in [−1, 1]. Forced mates
     count as ±1.
   - **Policy target:** the move the human played.
   - **Also stored:** Stockfish's best move and the game result, so the targets can be changed without
     relabelling.
3. **Network** (`model.py`). An AlphaZero-style residual network (scaled down to 8 blocks × 96 channels,
   20 input planes):
   - a policy head over 4,168 moves;
   - a value head with a single tanh output in [−1, 1].

   Loss: policy cross-entropy + squared value error.
4. **Search** (`mcts/gumbel.py`). Gumbel AlphaZero:
   - Gumbel-top-k at the root, then sequential halving with the mctx visit schedule.
   - Unvisited moves get the mixed value v_mix (completed Q).
   - The policy target is the improved policy π' = softmax(logits + σ(Q)).
   - Q is mapped to [0, 1] by (q + 1)/2 with c_visit = 50, c_scale = 1 (the paper's Go/chess setting);
     per-node min-max is an option.
5. **Self-play** (`training/selfplay.py`, `train_selfplay.py`):
   - 64 games share one network batch, and each game advances on its own.
   - Every move uses a 16-simulation search with Gumbel noise and becomes a training position.
   - Training steps follow the data: each position is used about `target_reuse` = 6 times.
   - Optional, off by default: resignation calibrated on no-resign games, KataGo playout-cap
     randomisation, and visit-count sampling in the opening.
6. **Evaluation** (`evaluate.py`, and inside the training loop). Colour-paired games against Stockfish
   (UCI_Elo) or another checkpoint, with an Elo estimate and 95% CI. During training, every 10 iterations
   the opponent is set to the previous estimate.

## Layout
| File | Purpose |
|---|---|
| `config.py` | All parameters: paths, model, MCTS, self-play, training, warm start, evaluation |
| `chess_env/encoding.py` | 20-plane board encoding, 4168-move policy index, game result as a value (side-to-move perspective) |
| `model.py` | `ChessNet` (residual tower, policy head, tanh value head), batched `Evaluator`, device auto-detection (xpu → cuda → mps → cpu), checkpoints |
| `mcts/gumbel.py` | Gumbel AlphaZero search as Python generators; `run_batched` batches leaf evaluations across searches |
| `training/selfplay.py` | Self-play games sharing one batch; Gumbel noise every move; optional opening exploration, playout-cap randomisation and resignation |
| `training/resign.py` | Resignation threshold calibrated on no-resign games (≤ 5% false positives, AlphaGo Zero); starts suspended and stays suspended while no threshold is safe |
| `training/replay_buffer.py` | Compact ring buffer (uint8 planes, top-128 sparse policy targets, float16 values) |
| `training/losses.py` | One training step: policy cross-entropy + value MSE |
| `training/logger.py` | JSONL + TensorBoard logging (per step, per iteration, per evaluation) |
| `download.py` | Downloads Stockfish and Lichess Elite PGNs |
| `warmstart.py` | `build`: sample positions and label them with Stockfish; `train`: supervised warm start |
| `train_selfplay.py` | Main loop: self-play → buffer → train → checkpoint → periodic Elo evaluation (exactly resumable) |
| `evaluate.py` | Parallel, colour-paired matches vs. Stockfish or checkpoints; Elo with 95% CI; move timing |
| `benchmark.py` | CPU vs. GPU throughput (inference, self-play, training) |
| `tests/` | Encoding, search (visit schedule, Q mapping, mate-in-one), resignation, self-play, paired match, train-loop resume |

## Workflow (run from `code/`)
```
pip install -r requirements.txt
python -m pytest tests
python download.py stockfish
python download.py elite          # months set in DownloadConfig.elite_months
python warmstart.py build         # positions + Stockfish labels -> data/warmstart_sf.npz
python warmstart.py train         # -> runs/v2/warmstart.pt
python train_selfplay.py          # runs until stopped; Ctrl+C and rerun to resume
python evaluate.py stockfish ../runs/v2/latest.pt
python evaluate.py match ../runs/v2/checkpoints/iter_00100.pt ../runs/v2/checkpoints/iter_00010.pt
```
`warmstart.py` takes overrides after the command, e.g. `python warmstart.py build warmstart.max_games=500`.
The 27k-game dataset (about 270k positions) takes about 70 minutes to label on an 8-core laptop.

Short test runs that write only to `test_run/<name>/`:
```
python ../test_run/run_test.py my_test mcts.n_simulations=32 mcts.c_scale=0.1
```

## Logs and TensorBoard
Each run directory `runs/<run_name>/` gets:
- `steps.jsonl`: every gradient step (loss, policy and value loss, policy accuracy, value MAE, gradient norm, lr)
- `log.jsonl`: every iteration (games and terminations, search-target entropy and KL to the prior,
  value error, resignation threshold and false-positive rate, buffer reuse, timings)
- `eval.jsonl`: every `TrainEvalConfig.every` iterations: Elo vs. Stockfish, a match vs. the previous
  snapshot, and seconds per move with the raw policy and with search
- `tensorboard/`: the same numbers. Start with `tensorboard --logdir runs` and open http://localhost:6006.

The warm start logs the same way to `runs/<run_name>/warmstart_logs/`: every gradient step, validation
metrics per epoch, and a match against Stockfish after each epoch (`WarmStartConfig.eval_*`: 30 games vs.
UCI_Elo 1500 at 0.1 s/move by default). It saves `warmstart.pt` and `warmstart_epochN.pt` after every epoch.

`latest.pt` and `buffer.npz` are written together every `checkpoint_every` iterations, with the RNG
states, so a rerun resumes exactly from the last checkpoint.

## Notes from the test runs (November 2026)
These used the earlier win/draw/loss value head.
- **Min-max Q with c_scale = 1 hurts.** It gave near one-hot policy targets and cost about 200 Elo even
  before training, so the default is the paper's (q + 1)/2 map.
- **Resignation is off by default.** Even at a value of −0.99, about 20% of would-be resignations did not
  lose, because the warm-start value head was overconfident.
- **Self-play from the old warm start lost strength within 10 iterations at every setting tried.** This is
  why the warm start was redone with Stockfish value targets and a scalar value head.

`runs/main/` holds the old win/draw/loss model, which does not load into the current network.

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
- Silver et al., *Mastering the game of Go without human knowledge*, Nature 2017 (resignation calibration).
- Wu, *Accelerating self-play learning in Go* (KataGo), 2019 (playout-cap randomisation).
- The Stockfish developers, *Stockfish 19* (value labels and evaluation opponent).
