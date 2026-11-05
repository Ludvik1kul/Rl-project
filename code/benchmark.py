"""Compare devices for inference (self-play) and training throughput.

Usage:
    python benchmark.py            # benchmarks cpu, and xpu / mps if available
"""
import time

import numpy as np
import torch

from config import Config
from model import ChessNet, Evaluator
from training.losses import train_step
from training.selfplay import play_games


def bench(device: str, cfg: Config):
    dev = torch.device(device)
    net = ChessNet(cfg.model)
    ev = Evaluator(net, dev)
    x = np.random.rand(cfg.selfplay.n_parallel_games, cfg.model.in_planes, 8, 8).astype(np.float32)
    for _ in range(5):
        ev(x)
    t = time.time()
    for _ in range(50):
        ev(x)
    infer = 50 * len(x) / (time.time() - t)

    cfg.selfplay.max_plies = 30
    t = time.time()
    play_games(ev, cfg, cfg.selfplay.n_parallel_games, np.random.default_rng(0))
    moves = cfg.selfplay.n_parallel_games * 30 / (time.time() - t)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    bs = cfg.train.batch_size
    planes = np.random.rand(bs, cfg.model.in_planes, 8, 8).astype(np.float32)
    moves_idx = np.random.randint(0, cfg.model.n_moves, bs)
    wdl = np.random.randint(0, 3, bs)
    for _ in range(3):
        train_step(net, opt, planes, moves_idx, wdl, dev)
    t = time.time()
    for _ in range(20):
        train_step(net, opt, planes, moves_idx, wdl, dev)
    steps = 20 / (time.time() - t)
    print(f"{device}: inference {infer:.0f} pos/s | self-play {moves:.0f} moves/s | train {steps:.2f} steps/s (bs {bs})")


if __name__ == "__main__":
    cfg = Config()
    bench("cpu", cfg)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        bench("xpu", Config())
    if torch.backends.mps.is_available():
        bench("mps", Config())
