"""Supervised warm start on Lichess Elite games.

Usage:
    python warmstart.py build   # PGN -> data/warmstart.npz
    python warmstart.py train   # trains the net, writes runs/<run_name>/warmstart.pt
"""
import sys
import time

import chess.pgn
import numpy as np
import torch

from chess_env.encoding import encode_board, encode_move, outcome_wdl
from config import Config
from model import ChessNet, get_device, save_checkpoint
from training.losses import train_step
from training.replay_buffer import compress_planes, decompress_planes


def build_dataset(cfg: Config):
    ws = cfg.warmstart
    planes, moves, wdl = [], [], []
    n_games = 0
    t0 = time.time()
    for pgn_path in sorted(cfg.paths.warmstart_pgn_dir.glob("*.pgn")):
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while n_games < ws.max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                h = game.headers
                result = h.get("Result", "*")
                try:
                    elos = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
                except ValueError:
                    continue
                if result not in ("1-0", "0-1", "1/2-1/2") or min(elos) < ws.min_elo:
                    continue
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= ws.skip_first_plies:
                        planes.append(compress_planes(encode_board(board)))
                        moves.append(encode_move(move, board.turn))
                        wdl.append(outcome_wdl(result, board.turn))
                    board.push(move)
                n_games += 1
                if n_games % 5000 == 0:
                    print(f"{n_games} games, {len(moves)} positions, {time.time() - t0:.0f}s")
        if n_games >= ws.max_games:
            break
    cfg.paths.warmstart_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.paths.warmstart_dataset, planes=np.stack(planes),
             moves=np.array(moves, dtype=np.int16), wdl=np.array(wdl, dtype=np.int8))
    print(f"Saved {len(moves)} positions from {n_games} games")


def train(cfg: Config):
    ws = cfg.warmstart
    rng = np.random.default_rng(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    d = np.load(cfg.paths.warmstart_dataset)
    planes, moves, wdl = d["planes"], d["moves"], d["wdl"]
    perm = rng.permutation(len(moves))
    n_val = int(len(perm) * ws.val_fraction)
    val, tr = perm[:n_val], perm[n_val:]

    device = get_device(cfg.device)
    net = ChessNet(cfg.model).to(device)
    steps_per_epoch = len(tr) // ws.batch_size
    opt = torch.optim.AdamW(net.parameters(), lr=ws.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, ws.lr, total_steps=ws.epochs * steps_per_epoch)
    print(f"Device {device}, {len(tr)} train / {n_val} val positions")

    for epoch in range(ws.epochs):
        rng.shuffle(tr)
        for s in range(steps_per_epoch):
            ix = np.sort(tr[s * ws.batch_size:(s + 1) * ws.batch_size])
            m = train_step(net, opt, decompress_planes(planes[ix]), moves[ix].astype(np.int64),
                           wdl[ix], device, cfg.train.value_loss_weight)
            sched.step()
            if s % 200 == 0:
                print(f"epoch {epoch} step {s}/{steps_per_epoch} " +
                      " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        print(f"epoch {epoch} val: {evaluate(net, planes, moves, wdl, val, device)}")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(cfg.run_dir / "warmstart.pt", net)
    print(f"Saved {cfg.run_dir / 'warmstart.pt'}")


@torch.inference_mode()
def evaluate(net, planes, moves, wdl, ix, device, bs=2048):
    net.eval()
    correct = vcorrect = 0
    for i in range(0, len(ix), bs):
        b = np.sort(ix[i:i + bs])
        logits, wl = net(torch.from_numpy(decompress_planes(planes[b])).to(device))
        correct += (logits.argmax(-1).cpu().numpy() == moves[b]).sum()
        vcorrect += (wl.argmax(-1).cpu().numpy() == wdl[b]).sum()
    return {"move_acc": correct / len(ix), "wdl_acc": vcorrect / len(ix)}


if __name__ == "__main__":
    cfg = Config()
    {"build": build_dataset, "train": train}[sys.argv[1]](cfg)
