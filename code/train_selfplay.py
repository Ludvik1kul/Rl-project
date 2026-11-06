"""Main self-play training loop: generate games -> replay buffer -> train -> checkpoint -> evaluate.

Resumes automatically from runs/<run_name>/latest.pt and buffer.npz, which are always written together
(with RNG states and the resignation calibration), so a resumed run continues where it was saved.
If no checkpoint exists, starts from runs/<run_name>/warmstart.pt (if present).
Logs: see training/logger.py (steps.jsonl, log.jsonl, eval.jsonl, tensorboard/).

Usage:
    python train_selfplay.py
"""
import os
import time
from collections import Counter

import chess
import numpy as np
import torch

from config import Config
from evaluate import AgentPlayer, StockfishPlayer, elo_estimate, play_match, time_moves
from model import ChessNet, Evaluator, get_device, load_checkpoint, save_checkpoint
from training.logger import RunLogger
from training.losses import dense_policy_target, train_step
from training.replay_buffer import ReplayBuffer
from training.resign import ResignCalibrator, trigger_level
from training.selfplay import play_games


def _atomic_save(path, write):
    """write(file) to a temporary file, then rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def play_iteration(evaluator, cfg: Config, rng, threshold):
    """Plays batches of games until min_positions_per_iteration is reached (at least one batch)."""
    samples, stats = [], []
    while True:
        s, st = play_games(evaluator, cfg, cfg.selfplay.games_per_iteration, rng, threshold)
        stats += st
        if s is not None:
            samples.append(s)
        if sum(len(x[3]) for x in samples) >= cfg.train.min_positions_per_iteration:
            break
    if not samples:
        return None, stats
    return tuple(np.concatenate([x[i] for x in samples]) for i in range(4)), stats


def selfplay_metrics(stats, calibrator):
    """Iteration statistics from the finished games (keys with '/' become TensorBoard groups)."""
    results = Counter(s["result"] for s in stats)
    terms = Counter(s["termination"] for s in stats)
    search = [x for s in stats for x in s["search"]]
    fp, fp_n = calibrator.false_positive_rate(calibrator.threshold)
    m = {
        "games/n": len(stats), "games/mean_plies": _mean([s["plies"] for s in stats]),
        "games/white_wins": results["1-0"], "games/black_wins": results["0-1"], "games/draws": results["1/2-1/2"],
        "search/target_entropy": _mean([e for e, _, _ in search]),
        "search/kl_target_prior": _mean([kl for _, kl, _ in search]),
        "search/value_mse_vs_result": _mean([se for _, _, se in search]),
        "resign/threshold": calibrator.threshold, "resign/false_positive_rate": fp, "resign/fp_samples": fp_n,
        "resign/no_resign_games": sum(not s["resign_allowed"] for s in stats),
    }
    for t in ("checkmate", "stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition",
              "fifty_moves", "threefold_repetition", "max_plies", "resign"):
        m[f"termination/{t}"] = terms[t]
    return m


def evaluate_strength(cfg: Config, net, device, iteration, elo_state, run, rng):
    """Match vs Stockfish at the previous Elo estimate, match vs the previous snapshot, move timing."""
    te = cfg.train_eval
    lo_b, hi_b = te.elo_bounds
    opp = int(np.clip(elo_state["next_opponent"], lo_b, hi_b))
    agent = AgentPlayer(cfg, net, te.sims, device)
    t0 = time.time()
    sf = StockfishPlayer(cfg, elo=opp, move_time=te.opponent_time)
    try:
        w, d, l = play_match(agent, sf, cfg, rng, te.n_games, verbose=False)
    finally:
        sf.close()
    diff, lo, hi = elo_estimate(w, d, l)
    entry = {
        "iteration": iteration, "opponent_elo": opp, "wins": w, "draws": d, "losses": l,
        "score": (w + 0.5 * d) / (w + d + l), "elo": opp + diff, "elo_lo": opp + lo, "elo_hi": opp + hi,
        "timing/agent_sec_per_move_batched": agent.search_time / max(agent.n_moves, 1),
        "timing/stockfish_sec_per_move": sf.search_time / max(sf.n_moves, 1),
        "timing/sf_match_sec": time.time() - t0,
    }
    # The opponent follows the estimate, at most max_elo_step per evaluation.
    elo_state["estimate"] = opp + diff
    elo_state["next_opponent"] = int(np.clip(opp + diff, opp - te.max_elo_step, opp + te.max_elo_step))

    prev = run / "checkpoints" / f"iter_{iteration - te.every:05d}.pt"
    if te.match_games and prev.exists():
        t0 = time.time()
        mw, md, ml = play_match(agent, AgentPlayer(cfg, prev, te.sims, device), cfg, rng, te.match_games,
                                verbose=False)
        entry |= {"match/vs_iteration": iteration - te.every, "match/wins": mw, "match/draws": md,
                  "match/losses": ml, "match/elo_diff": elo_estimate(mw, md, ml)[0],
                  "timing/match_sec": time.time() - t0}

    boards = [b for _, b in zip(range(te.timing_positions), _timing_boards(rng))]
    t_policy, t_search = time_moves(agent, boards, te.sims)
    entry |= {"timing/policy_sec_per_move": t_policy, "timing/search_sec_per_move": t_search}
    return entry


def _timing_boards(rng):
    """Positions for move timing: openings of 8-40 random plies (only the move count matters here)."""
    while True:
        b = chess.Board()
        for _ in range(int(rng.integers(8, 40))):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            yield b


def main(cfg: Config):
    tc, te = cfg.train, cfg.train_eval
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    (run / "checkpoints").mkdir(exist_ok=True)
    rng = np.random.default_rng(tc.seed)
    torch.manual_seed(tc.seed)

    device = get_device(cfg.device)
    net = ChessNet(cfg.model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    buffer = ReplayBuffer(tc.buffer_size)
    calibrator = ResignCalibrator(cfg.selfplay)
    elo_state = {"estimate": None, "next_opponent": te.initial_elo}
    iteration = global_step = 0

    if (run / "latest.pt").exists():
        state = load_checkpoint(run / "latest.pt", net, opt, map_location=device)
        iteration = state["iteration"]
        global_step = state.get("global_step", 0)
        if "rng" in state:
            rng.bit_generator.state = state["rng"]
            torch.set_rng_state(state["torch_rng"])
        if "resign" in state:
            calibrator.load_state_dict(state["resign"])
        elo_state = state.get("elo", elo_state)
        if (run / "buffer.npz").exists():
            buffer.load(run / "buffer.npz")
        print(f"Resumed at iteration {iteration}, buffer {buffer.size}")
    elif (run / "warmstart.pt").exists():
        load_checkpoint(run / "warmstart.pt", net, map_location=device)
        print("Starting from warm-start weights")
    else:
        print("Starting from random weights")

    def checkpoint():
        _atomic_save(run / "latest.pt", lambda f: save_checkpoint(
            f, net, opt, iteration=iteration, global_step=global_step, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), resign=calibrator.state_dict(), elo=elo_state))
        _atomic_save(run / "buffer.npz", buffer.save)

    def evaluate():
        entry = evaluate_strength(cfg, net, device, iteration, elo_state, run,
                                  np.random.default_rng([tc.seed, iteration]))
        logger.write("eval", iteration, entry)
        print(f"[eval {iteration}] elo {entry['elo']:.0f} [{entry['elo_lo']:.0f}, {entry['elo_hi']:.0f}] "
              f"vs SF {entry['opponent_elo']} W/D/L {entry['wins']}/{entry['draws']}/{entry['losses']}")

    logger = RunLogger(run)
    evaluator = Evaluator(net, device)
    if iteration == 0:
        save_checkpoint(run / "checkpoints" / "iter_00000.pt", net, iteration=0)
        if te.every and te.at_start:
            evaluate()
    t_start = time.time()
    while iteration < tc.n_iterations:
        t0 = time.time()
        calls0, pos0 = evaluator.n_calls, evaluator.n_positions
        samples, stats = play_iteration(evaluator, cfg, rng, calibrator.threshold)
        new_positions = 0 if samples is None else len(samples[3])
        if samples is not None:
            buffer.add(*samples)
        t_play = time.time() - t0
        k = cfg.selfplay.resign_consecutive
        for s in stats:
            if not s["resign_allowed"]:
                for side, values in s["values"].items():
                    calibrator.add(trigger_level(values, k), s["lost"][side])
        calibrator.update()

        losses = []
        if buffer.size >= tc.min_buffer_size:
            n_steps = tc.train_steps_per_iteration if tc.target_reuse <= 0 else \
                int(np.clip(round(tc.target_reuse * new_positions / tc.batch_size), 1, tc.max_train_steps_per_iteration))
            for _ in range(n_steps):
                planes, pi, pp, wdl = buffer.sample(tc.batch_size, rng)
                target = dense_policy_target(pi, pp, cfg.model.n_moves)
                m = train_step(net, opt, planes, target, wdl, device, tc.value_loss_weight)
                global_step += 1
                losses.append(m)
                logger.write("steps", global_step, m | {"lr": opt.param_groups[0]["lr"], "iteration": iteration + 1},
                             tb_prefix="train")
        iteration += 1
        t_train = time.time() - t0 - t_play

        plies = sum(s["plies"] for s in stats)
        entry = {
            "iteration": iteration, "global_step": global_step,
            "timing/play_sec": t_play, "timing/train_sec": t_train,
            "timing/selfplay_sec_per_move": t_play / max(plies, 1),
            "timing/selfplay_moves_per_sec": plies / max(t_play, 1e-9),
            "timing/nn_positions_per_sec": (evaluator.n_positions - pos0) / max(t_play, 1e-9),
            "timing/mean_nn_batch": (evaluator.n_positions - pos0) / max(evaluator.n_calls - calls0, 1),
            "buffer/new_positions": new_positions, "buffer/size": buffer.size,
            "buffer/total_positions": buffer.total_added, "buffer/train_steps": len(losses),
            "buffer/reuse": len(losses) * tc.batch_size / max(new_positions, 1),
        } | selfplay_metrics(stats, calibrator)
        if losses:
            entry |= {f"train_mean/{k}": _mean([l[k] for l in losses]) for k in losses[0]}
        logger.write("log", iteration, entry)
        print(f"iter {iteration}: {len(stats)} games, {new_positions} positions, {len(losses)} steps, "
              f"loss {entry.get('train_mean/loss', float('nan')):.3f}, play {t_play:.0f}s, train {t_train:.0f}s")

        out_of_time = tc.time_limit_hours and time.time() - t_start > tc.time_limit_hours * 3600
        if iteration % tc.snapshot_every == 0:
            save_checkpoint(run / "checkpoints" / f"iter_{iteration:05d}.pt", net, iteration=iteration)
        if te.every and iteration % te.every == 0:
            evaluate()
        if iteration % tc.checkpoint_every == 0 or out_of_time:
            checkpoint()
        if out_of_time:
            print(f"Time limit reached after iteration {iteration}")
            break
    logger.close()


if __name__ == "__main__":
    main(Config())
