"""Main self-play training loop: generate games -> replay buffer -> train -> checkpoint.

Resumes automatically from runs/<run_name>/latest.pt and buffer.npz.
If no checkpoint exists, starts from runs/<run_name>/warmstart.pt (if present).

Usage:
    python train_selfplay.py
"""
import json
import time

import numpy as np
import torch

from config import Config
from model import ChessNet, Evaluator, get_device, load_checkpoint, save_checkpoint
from training.losses import dense_policy_target, train_step
from training.replay_buffer import ReplayBuffer
from training.selfplay import play_games


def main(cfg: Config):
    tc = cfg.train
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    rng = np.random.default_rng(tc.seed)
    torch.manual_seed(tc.seed)

    device = get_device(cfg.device)
    net = ChessNet(cfg.model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    buffer = ReplayBuffer(tc.buffer_size)
    iteration = 0

    if (run / "latest.pt").exists():
        state = load_checkpoint(run / "latest.pt", net, opt, map_location=device)
        iteration = state["iteration"]
        if (run / "buffer.npz").exists():
            buffer.load(run / "buffer.npz")
        print(f"Resumed at iteration {iteration}, buffer {buffer.size}")
    elif (run / "warmstart.pt").exists():
        load_checkpoint(run / "warmstart.pt", net, map_location=device)
        print("Starting from warm-start weights")
    else:
        print("Starting from random weights")

    evaluator = Evaluator(net, device)
    log = open(run / "log.jsonl", "a")
    while iteration < tc.n_iterations:
        t0 = time.time()
        samples, stats = play_games(evaluator, cfg, cfg.selfplay.games_per_iteration, rng)
        if samples is not None:
            buffer.add(*samples)
        t_play = time.time() - t0

        losses = []
        if buffer.size >= tc.min_buffer_size:
            for _ in range(tc.train_steps_per_iteration):
                planes, pi, pp, wdl = buffer.sample(tc.batch_size, rng)
                target = dense_policy_target(pi, pp, cfg.model.n_moves)
                losses.append(train_step(net, opt, planes, target, wdl, device, tc.value_loss_weight))
        iteration += 1

        results = [s["result"] for s in stats]
        entry = {
            "iteration": iteration, "time": time.time(), "play_sec": t_play,
            "train_sec": time.time() - t0 - t_play, "games": len(stats),
            "positions": 0 if samples is None else len(samples[3]), "buffer": buffer.size,
            "total_positions": buffer.total_added,
            "white_wins": results.count("1-0"), "black_wins": results.count("0-1"),
            "draws": results.count("1/2-1/2"), "mean_plies": float(np.mean([s["plies"] for s in stats])),
        }
        if losses:
            for k in losses[0]:
                entry[k] = float(np.mean([l[k] for l in losses]))
        print(json.dumps(entry))
        log.write(json.dumps(entry) + "\n"); log.flush()

        if iteration % tc.checkpoint_every == 0:
            save_checkpoint(run / "latest.pt", net, opt, iteration=iteration)
        if iteration % tc.snapshot_every == 0:
            save_checkpoint(run / "checkpoints" / f"iter_{iteration:05d}.pt", net, iteration=iteration)
        if iteration % tc.buffer_save_every == 0:
            buffer.save(run / "buffer.npz")


if __name__ == "__main__":
    main(Config())
