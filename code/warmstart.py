"""Supervised warm start on Lichess Elite games with Stockfish value labels.

Usage:
    python warmstart.py build   # PGN -> sampled positions -> Stockfish labels -> data/warmstart_sf.npz
    python warmstart.py train   # trains the net, writes runs/<run_name>/warmstart.pt
Optional overrides after the command, e.g. `python warmstart.py build warmstart.max_games=500`.

Targets: policy = the move the human played; value = Stockfish's own win/draw/loss estimate for the
side to move, as P(win) - P(loss) in [-1, 1] (forced mates give +-1). Stockfish's best move and the game
result are stored as well, so the policy target or a value blend can change without relabelling.
"""
import multiprocessing as mp
import sys
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm

from chess_env.encoding import encode_board, encode_move, outcome_value
from config import Config
from evaluate import AgentPlayer, StockfishPlayer, elo_estimate, play_match
from model import ChessNet, get_device, save_checkpoint
from training.logger import RunLogger
from training.losses import train_step
from training.replay_buffer import compress_planes, decompress_planes


def sample_positions(cfg: Config, rng: np.random.Generator):
    """
    Reads each .pgn file in ROOT / data / lichess_elite, one at time, with python-chess.

    It keeps a game only if it finished with a real result ("1-0", "0-1", "1/2-1/2") and
    both players are rated at least cfg.warmstart.min_elo. It stops after cfg.warmstart.max_games.

    From each kept game it draws up to cfg.warmstart.positions_per_game random positions (after
    cfg.warmstart.skip_first_plies), so the dataset covers more games with less correlated positions.

    Returns a list of (fen, human move index, game result value for the side to move, game id).

    Resources:
    python-chess doc: https://python-chess.readthedocs.io/en/latest/index.html
    """
    ws = cfg.warmstart
    out, n_games = [], 0
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
                moves = list(game.mainline_moves())
                plies = np.arange(ws.skip_first_plies, len(moves))
                if not len(plies):
                    continue
                chosen = set(rng.choice(plies, min(ws.positions_per_game, len(plies)), replace=False).tolist())

                # Replay the game (already filtered) from scratch.
                # See https://python-chess.readthedocs.io/en/latest/_modules/chess/pgn.html#GameNode.board
                board = game.board()
                for ply, move in enumerate(moves):
                    if ply in chosen:
                        out.append((board.fen(), encode_move(move, board.turn), outcome_value(result, board.turn),
                                    n_games))
                    board.push(move)
                n_games += 1
        if n_games >= ws.max_games:
            break
    print(f"Sampled {len(out)} positions from {n_games} games")
    return out


# ---------------------------------------------------------------- Stockfish labelling (worker processes)
_engine = _limit = None


def _init_worker(path: str, nodes: int, hash_mb: int):
    global _engine, _limit
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})
    _limit = chess.engine.Limit(nodes=nodes)


def _label(fen: str):
    """(value for the side to move in [-1, 1], best move index or -1)."""
    board = chess.Board(fen)
    info = _engine.analyse(board, _limit)
    score = info["score"].pov(board.turn)
    wdl = info["wdl"].pov(board.turn) if "wdl" in info else score.wdl(ply=board.ply())
    value = (wdl.wins - wdl.losses) / max(wdl.total(), 1)
    if score.is_mate():
        value = 1.0 if score.mate() > 0 else -1.0
    pv = info.get("pv")
    return float(value), encode_move(pv[0], board.turn) if pv else -1


def label_positions(cfg: Config, fens):
    ws = cfg.warmstart
    args = (str(cfg.paths.stockfish_path), ws.sf_nodes, ws.sf_hash_mb)
    with mp.Pool(ws.sf_workers, initializer=_init_worker, initargs=args) as pool:
        return list(tqdm(pool.imap(_label, fens, chunksize=32), total=len(fens), desc="stockfish",
                         dynamic_ncols=True))


def build_dataset(cfg: Config):
    """
    Samples positions (sample_positions), labels them with Stockfish (label_positions) and saves
    cfg.paths.warmstart_dataset (data/warmstart_sf.npz) with, per position:
    - planes    :   uint8, chess_env.encoding.encode_board compressed by replay_buffer.compress_planes
    - moves     :   int16, the human move (chess_env.encoding.encode_move), the policy target
    - sf_value  :   float32, Stockfish's P(win) - P(loss) for the side to move, the value target
    - sf_move   :   int16, Stockfish's best move (-1 if none)
    - result    :   float32, the game result for the side to move (+1 / 0 / -1)
    - game      :   int32, the game id, used later for the validation split
    """
    ws = cfg.warmstart
    t0 = time.time()
    samples = sample_positions(cfg, np.random.default_rng(cfg.train.seed))
    t_sample = time.time() - t0
    labels = label_positions(cfg, [s[0] for s in samples])
    t_label = time.time() - t0 - t_sample
    planes = np.stack([compress_planes(encode_board(chess.Board(s[0]))) for s in samples])
    cfg.paths.warmstart_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.paths.warmstart_dataset, planes=planes,
             moves=np.array([s[1] for s in samples], dtype=np.int16),
             result=np.array([s[2] for s in samples], dtype=np.float32),
             game=np.array([s[3] for s in samples], dtype=np.int32),
             sf_value=np.array([l[0] for l in labels], dtype=np.float32),
             sf_move=np.array([l[1] for l in labels], dtype=np.int16),
             sf_nodes=ws.sf_nodes)
    v = np.array([l[0] for l in labels])
    print(f"Saved {len(samples)} positions to {cfg.paths.warmstart_dataset} "
          f"(sampling {t_sample / 60:.1f} min, Stockfish {t_label / 60:.1f} min, "
          f"{len(samples) / max(t_label, 1e-9):.0f} positions/s)")
    print(f"Stockfish value: mean {v.mean():+.3f}, |v| > 0.9: {np.mean(np.abs(v) > 0.9):.1%}, "
          f"|v| < 0.1: {np.mean(np.abs(v) < 0.1):.1%}")


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
        - value head with 1 tanh output in [-1, 1]

    Trained with AdamW optimizer with a OneCycle scheduler for a dynamic learning rate.
    Learning rate in cfg.warmstart.lr (default at 2e-3)
    Learning rate decay in cfg.train.weight_decay (default at 1e-4).

    At each epoch, it shuffles the training indices and walks trough them in batches cfg.warmstart.batch_size

    Logging (runs/<run_name>/warmstart_logs, see training/logger.py; TensorBoard in its tensorboard/):
        - steps.jsonl   every gradient step: losses, accuracies, gradient norm, lr
        - log.jsonl     every epoch: validation metrics (see evaluate)
        - eval.jsonl    every epoch: match against Stockfish (cfg.warmstart.eval_*), Elo with 95% CI
    The weights are saved after every epoch (warmstart.pt, plus warmstart_epochN.pt).

    At each step, losses.train_step is called:
        - policy loss   = cross-entropy against the human move;
        - value loss    = mean squared error against the value target, which is
                          cfg.warmstart.sf_value_weight * Stockfish value + (1 - sf_value_weight) * game result;
        - loss          = policy loss + value_loss_weight * value loss (cfg.train.value_loss_weight, default 1)


    Resources:
    - https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.OneCycleLR.html
    """
    ws = cfg.warmstart
    rng = np.random.default_rng(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    d = np.load(cfg.paths.warmstart_dataset)
    planes, moves = d["planes"], d["moves"]
    value = (ws.sf_value_weight * d["sf_value"] + (1 - ws.sf_value_weight) * d["result"]).astype(np.float32)
    val, tr = split_by_game(d, len(moves), ws.val_fraction, rng)

    device = get_device(cfg.device)
    net = ChessNet(cfg.model).to(device)
    steps_per_epoch = len(tr) // ws.batch_size
    # NOTE: AdamW optimizer with a OneCycle scheduler for a dynamic learning rate
    # See docs: https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.OneCycleLR.html
    opt = torch.optim.AdamW(net.parameters(), lr=ws.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, ws.lr, total_steps=ws.epochs * steps_per_epoch)
    print(f"Device {device}, {len(tr)} train / {len(val)} val positions, value target "
          f"{ws.sf_value_weight:g} x Stockfish + {1 - ws.sf_value_weight:g} x result")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.run_dir / "warmstart_logs"
    log_dir.mkdir(exist_ok=True)
    logger = RunLogger(log_dir)
    step = 0

    for epoch in range(ws.epochs):
        rng.shuffle(tr)
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{ws.epochs}", dynamic_ncols=True)
        for s in bar:
            # NOTE: within a batch, indices are sorted only to speed up memory reads
            ix = np.sort(tr[s * ws.batch_size:(s + 1) * ws.batch_size])
            m = train_step(net, opt, decompress_planes(planes[ix]), moves[ix].astype(np.int64),
                           value[ix], device, cfg.train.value_loss_weight)
            step += 1
            logger.write("steps", step, m | {"lr": sched.get_last_lr()[0], "epoch": epoch + 1}, tb_prefix="train")
            sched.step()
            if s % 20 == 0:
                bar.set_postfix({k: f"{v:.3f}" for k, v in m.items()} | {"lr": f"{sched.get_last_lr()[0]:.1e}"})
        metrics = evaluate(net, planes, moves, d['sf_move'], value, val, device)
        tqdm.write(f"epoch {epoch + 1} val: {metrics}")
        logger.write("log", epoch + 1, {f"val/{k}": v for k, v in metrics.items()} | {"global_step": step})
        save_checkpoint(cfg.run_dir / "warmstart.pt", net, epoch=epoch + 1)
        save_checkpoint(cfg.run_dir / f"warmstart_epoch{epoch + 1}.pt", net, epoch=epoch + 1)
        if ws.eval_games:
            entry = match_stockfish(cfg, net, device, np.random.default_rng([cfg.train.seed, epoch]))
            logger.write("eval", epoch + 1, entry | {"global_step": step})
            tqdm.write(f"epoch {epoch + 1} vs Stockfish {ws.eval_elo}: W/D/L {entry['wins']}/{entry['draws']}/"
                       f"{entry['losses']}, Elo {entry['elo']:.0f} [{entry['elo_lo']:.0f}, {entry['elo_hi']:.0f}]")
    logger.close()
    print(f"Saved {cfg.run_dir / 'warmstart.pt'}")


def match_stockfish(cfg: Config, net, device, rng):
    """cfg.warmstart.eval_games colour-paired games against Stockfish at UCI_Elo eval_elo (evaluate.play_match)."""
    ws = cfg.warmstart
    agent = AgentPlayer(cfg, net, ws.eval_sims, device)
    t0 = time.time()
    sf = StockfishPlayer(cfg, elo=ws.eval_elo, move_time=ws.eval_time)
    try:
        w, d, l = play_match(agent, sf, cfg, rng, ws.eval_games, verbose=False)
    finally:
        sf.close()
    diff, lo, hi = elo_estimate(w, d, l)
    return {"opponent_elo": ws.eval_elo, "wins": w, "draws": d, "losses": l, "score": (w + 0.5 * d) / (w + d + l),
            "elo": ws.eval_elo + diff, "elo_lo": ws.eval_elo + lo, "elo_hi": ws.eval_elo + hi,
            "match_sec": time.time() - t0}


@torch.inference_mode()  # no autograd graph: faster, less memory, no accidental training
def evaluate(net, planes, moves, sf_moves, value, ix, device, bs=2048):
    """Validation metrics, averaged per position: policy cross-entropy against the human move, value MSE
    (the same losses as train_step), move accuracy for the human and the Stockfish move, value MAE, and
    the value MSE of always predicting the mean (a baseline)."""
    net.eval()
    human = sf = se = ae = ce = 0.0
    for i in tqdm(range(0, len(ix), bs), desc="val", leave=False, dynamic_ncols=True):
        b = np.sort(ix[i:i + bs])
        logits, v = net(torch.from_numpy(decompress_planes(planes[b])).to(device))
        logp = torch.log_softmax(logits.float(), -1)
        ce -= logp.gather(1, torch.from_numpy(moves[b].astype(np.int64)).to(device)[:, None]).sum().item()
        top = logits.argmax(-1).cpu().numpy()
        human += (top == moves[b]).sum()
        sf += (top == sf_moves[b]).sum()
        err = v.float().cpu().numpy() - value[b]
        se += (err ** 2).sum()
        ae += np.abs(err).sum()
    n = len(ix)
    metrics = {"policy_loss": ce / n, "move_acc": human / n, "sf_move_acc": sf / n, "value_mse": se / n,
               "value_mae": ae / n, "value_mse_const": value[ix].var()}
    return {k: round(float(v), 4) for k, v in metrics.items()}


def apply_overrides(cfg: Config, args):
    for arg in args:
        key, val = arg.split("=", 1)
        section, field = key.split(".")
        obj = getattr(cfg, section)
        setattr(obj, field, type(getattr(obj, field))(val))
        print(f"{key} = {getattr(obj, field)!r}")


if __name__ == "__main__":
    cfg = Config()
    apply_overrides(cfg, sys.argv[2:])
    {"build": build_dataset, "train": train}[sys.argv[1]](cfg)
