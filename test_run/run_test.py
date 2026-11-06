"""One-hour smoke run of the full self-play pipeline (training, logging, Elo evaluation).

Everything is written inside test_run/ (a copy of warmstart.pt is in test_run/run/), so this folder
can be deleted afterwards. runs/main is never touched.

    cd code
    ../.venv-xpu/Scripts/python.exe ../test_run/run_test.py
    ../.venv-xpu/Scripts/tensorboard.exe --logdir ../test_run/run/tensorboard
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from config import Config  # noqa: E402
import train_selfplay  # noqa: E402

cfg = Config()
cfg.paths.runs_dir = HERE          # run dir = test_run/run (holds the warmstart.pt copy)
cfg.run_name = "run"

# Scaled down so several evaluations fit in an hour; numbers are noisy by design.
cfg.selfplay.games_per_iteration = 64
cfg.selfplay.n_parallel_games = 64
cfg.train.buffer_size = 50_000
cfg.train.min_buffer_size = 5_000
cfg.train.checkpoint_every = 5
cfg.train.snapshot_every = 10
cfg.train.time_limit_hours = 40 / 60
cfg.train_eval.every = 10
cfg.train_eval.n_games = 20
cfg.train_eval.match_games = 10
cfg.train_eval.timing_positions = 30

if __name__ == "__main__":
    train_selfplay.main(cfg)
