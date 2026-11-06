"""Short smoke run of the full self-play pipeline (training, logging, Elo evaluation).

Everything is written inside test_run/<run name>/ (starting from a copy of warmstart.pt), so this folder
can be deleted afterwards. runs/main is never touched.

    cd code
    ../.venv-xpu/Scripts/python.exe ../test_run/run_test.py                       # run "run"
    ../.venv-xpu/Scripts/python.exe ../test_run/run_test.py run2 mcts.n_simulations=32
    ../.venv-xpu/Scripts/tensorboard.exe --logdir ../test_run

Arguments: an optional run name, then any number of config overrides section.field=value.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from config import Config  # noqa: E402
import train_selfplay  # noqa: E402

cfg = Config()
cfg.paths.runs_dir = HERE          # run dir = test_run/<run name>
cfg.run_name = "run"

# Scaled down so several evaluations fit in the time limit; numbers are noisy by design.
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


def apply_overrides(cfg, args):
    for arg in args:
        if "=" not in arg:
            cfg.run_name = arg
            continue
        key, value = arg.split("=", 1)
        section, field = key.split(".")
        obj = getattr(cfg, section)
        old = getattr(obj, field)
        setattr(obj, field, value if isinstance(old, str) else type(old)(value))
        print(f"{key} = {getattr(obj, field)!r}")


if __name__ == "__main__":
    apply_overrides(cfg, sys.argv[1:])
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if not (cfg.run_dir / "warmstart.pt").exists():
        shutil.copy(HERE.parent / "runs" / "main" / "warmstart.pt", cfg.run_dir / "warmstart.pt")
    train_selfplay.main(cfg)
