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
from tqdm import tqdm

from chess_env.encoding import encode_board, encode_move, outcome_wdl
from config import Config
from model import ChessNet, get_device, save_checkpoint
from training.losses import train_step
from training.replay_buffer import compress_planes, decompress_planes


def build_dataset(cfg: Config):
    """
    Reads each .pgn file in ROOT / data / lichess_elite, one at time, with python-chess.
    
    It keeps a game only if it finished with a real results ("1-0", "0-1", "1/2-1/2") and
    both players are rated at least cfg.warmstart.min_elo. It stops after cfg.warmstart.max_games

    Then, for each game, it replay the match and for every position (move in the game), it saves:
    - planes :   first encoded to uint8 via chess_env.encoding.encode_board
    - move   :   also encoded via chess_env.encoding.encode_move
    - wdl    :   chess_env.encoding.outcome_wdl
    - game   :   the game id, used later for the validation split

    Finally everything is saved in ROOT / data /. In particular:
    - planes    in np.int8
    - moves     in np.int16
    - wdl       in np.int8
    - game      in np.int32

    Resources: 
    python-chess doc: https://python-chess.readthedocs.io/en/latest/index.html
    """
    ws = cfg.warmstart
    planes, moves, wdl, game_ids = [], [], [], []
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

                # Replay the game (already filtered) from scratch. 
                # See https://python-chess.readthedocs.io/en/latest/_modules/chess/pgn.html#GameNode.board
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= ws.skip_first_plies:
                        planes.append(compress_planes(encode_board(board)))
                        moves.append(encode_move(move, board.turn))
                        wdl.append(outcome_wdl(result, board.turn))
                        game_ids.append(n_games)
                    board.push(move)
                n_games += 1
                if n_games % 5000 == 0:
                    print(f"{n_games} games, {len(moves)} positions, {time.time() - t0:.0f}s")
        if n_games >= ws.max_games:
            break
    cfg.paths.warmstart_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.paths.warmstart_dataset, planes=np.stack(planes),
             moves=np.array(moves, dtype=np.int16), wdl=np.array(wdl, dtype=np.int8),
             game=np.array(game_ids, dtype=np.int32))
    print(f"Saved {len(moves)} positions from {n_games} games")


def split_by_game(d, n: int, val_fraction: float, rng: np.random.Generator):
    """(val, train) position indices with no game in both sets.

    Datasets built before game ids were stored keep positions in game order, so the last block of
    positions is used as validation (at most one game straddles the boundary).

    Hyperparameter for % of validation in cfg.warmstart.val_fraction
    """
    if "game" in d:
        game = d["game"]
        games = np.unique(game)
        val_games = rng.choice(games, int(round(len(games) * val_fraction)), replace=False)
        is_val = np.isin(game, val_games)
        return np.flatnonzero(is_val), np.flatnonzero(~is_val)
    n_val = int(n * val_fraction)
    return np.arange(n - n_val, n), np.arange(n - n_val)


def train(cfg: Config):
    """
    Builds a model.ChessNet:
        - 8 residual block and 96 channels, 
        - policy head with 4 168 logits
        - value head with 3 WDL logits
    
    Trained with AdamW optimizer with a OneCycle scheduler for a dynamic learning rate.
    Learning rate in cfg.warmstart.lr (default at 2e-3)
    Learning rate decay in cfg.train.weight_decay (default at 1e-4).

    At each epoch, it shuffles the training indices and walks trough them in batches cfg.warmstart.batch_size

    At each step, losses.train_step is called and it uses negative log likelihood loss:
        - policy loss   = cross-entropy against the human move;
        - value loss    = cross-entropy against W/D/L;
        - loss          = policy loss + values_loss_weigth * value loss (where values_loss_weight is cfg.train.value_loss_weight, default at 1)


    Resources: 
    - https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.OneCycleLR.html
    """
    ws = cfg.warmstart
    rng = np.random.default_rng(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    d = np.load(cfg.paths.warmstart_dataset)
    planes, moves, wdl = d["planes"], d["moves"], d["wdl"]
    val, tr = split_by_game(d, len(moves), ws.val_fraction, rng)
    n_val = len(val)

    device = get_device(cfg.device)
    net = ChessNet(cfg.model).to(device)
    steps_per_epoch = len(tr) // ws.batch_size
    # NOTE: AdamW optimizer with a OneCycle scheduler for a dynamic learning rate 
    # See docs: https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.OneCycleLR.html
    opt = torch.optim.AdamW(net.parameters(), lr=ws.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, ws.lr, total_steps=ws.epochs * steps_per_epoch)
    print(f"Device {device}, {len(tr)} train / {n_val} val positions")

    for epoch in range(ws.epochs):
        rng.shuffle(tr)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{ws.epochs}", dynamic_ncols=True)
        for s in bar:
            # NOTE: within a batch, indices are sorted only to speed up memory reads
            ix = np.sort(tr[s * ws.batch_size:(s + 1) * ws.batch_size])
            m = train_step(net, opt, decompress_planes(planes[ix]), moves[ix].astype(np.int64),
                           wdl[ix], device, cfg.train.value_loss_weight)
            sched.step()
            if s % 20 == 0:
                bar.set_postfix({k: f"{v:.3f}" for k, v in m.items()} | {"lr": f"{sched.get_last_lr()[0]:.1e}"})
        tqdm.write(f"epoch {epoch + 1} val: {evaluate(net, planes, moves, wdl, val, device)}")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(cfg.run_dir / "warmstart.pt", net)
    print(f"Saved {cfg.run_dir / 'warmstart.pt'}")


@torch.inference_mode()  # no autograd graph: faster, less memory, no accidental training
def evaluate(net, planes, moves, wdl, ix, device, bs=2048):
    net.eval()
    correct = vcorrect = 0
    for i in tqdm(range(0, len(ix), bs), desc="val", leave=False, dynamic_ncols=True):
        b = np.sort(ix[i:i + bs])
        logits, wl = net(torch.from_numpy(decompress_planes(planes[b])).to(device))
        correct += (logits.argmax(-1).cpu().numpy() == moves[b]).sum()
        vcorrect += (wl.argmax(-1).cpu().numpy() == wdl[b]).sum()
    return {"move_acc": correct / len(ix), "wdl_acc": vcorrect / len(ix)}


if __name__ == "__main__":
    cfg = Config()
    {"build": build_dataset, "train": train}[sys.argv[1]](cfg)
