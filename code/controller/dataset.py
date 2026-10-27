"""Counterfactual dataset: for positions from the base agent's games, the outcome of every action.

Usage:
    python -m controller.dataset ../runs/main/latest.pt
Writes runs/<run_name>/controller_data.npz
"""
import sys
import time

import chess
import chess.engine
import numpy as np

from config import Config
from controller.features import extract_features
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device, load_checkpoint


def oracle_cp(engine, board: chess.Board, move: chess.Move, cfg) -> float:
    """Oracle eval (centipawns, mover's perspective) of the position after `move`."""
    mover = board.turn
    board.push(move)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=cfg.oracle_depth))
        return float(info["score"].pov(mover).score(mate_score=cfg.mate_score))
    finally:
        board.pop()


def generate(cfg: Config, ckpt_path):
    cc = cfg.controller
    rng = np.random.default_rng(cfg.train.seed)
    device = get_device(cfg.device)
    net = ChessNet(cfg.model)
    load_checkpoint(ckpt_path, net, map_location=device)
    ev = Evaluator(net, device)
    oracle = chess.engine.SimpleEngine.popen_uci(str(cfg.paths.stockfish_path))
    oracle.configure({"Threads": cc.oracle_threads})
    expert = chess.engine.SimpleEngine.popen_uci(str(cfg.paths.stockfish_path))
    expert.configure({"UCI_LimitStrength": True, "UCI_Elo": cc.expert_elo})
    expert_limit = chess.engine.Limit(time=cc.expert_time)

    rows = {k: [] for k in ("features", "cp_base", "cp_deeper", "cp_expert", "cp_random",
                            "u", "game", "same_deeper", "same_expert", "fen")}
    t0 = time.time()
    try:
        for start in range(0, cc.n_dataset_games, cc.n_parallel_games):
            ids = list(range(start, min(start + cc.n_parallel_games, cc.n_dataset_games)))
            boards = []
            for _ in ids:
                b = chess.Board()
                for _ in range(cc.random_opening_plies):
                    b.push(rng.choice(list(b.legal_moves)))
                boards.append(b)
            prev_value = [{chess.WHITE: None, chess.BLACK: None} for _ in ids]
            active = [i for i, b in enumerate(boards) if not b.is_game_over()]
            while active:
                base = run_batched(ev, [gumbel_search(boards[i], cc.base_sims, cfg.mcts, 0.0, rng) for i in active])
                label = [i for i in active if rng.random() < cc.sample_prob]
                deeper = dict(zip(label, run_batched(ev, [
                    gumbel_search(boards[i], cc.base_sims * cc.deeper_multiplier, cfg.mcts, 0.0, rng)
                    for i in label])))
                nxt = []
                for i, res in zip(active, base):
                    b = boards[i]
                    if i in deeper:
                        mv_expert = expert.play(b, expert_limit).move
                        mv_random = rng.choice(list(b.legal_moves))
                        rows["features"].append(extract_features(b, res, prev_value[i][b.turn]))
                        rows["cp_base"].append(oracle_cp(oracle, b, res.move, cc))
                        rows["cp_deeper"].append(rows["cp_base"][-1] if deeper[i].move == res.move
                                                 else oracle_cp(oracle, b, deeper[i].move, cc))
                        rows["cp_expert"].append(rows["cp_base"][-1] if mv_expert == res.move
                                                 else oracle_cp(oracle, b, mv_expert, cc))
                        rows["cp_random"].append(oracle_cp(oracle, b, mv_random, cc))
                        rows["u"].append(rng.random())
                        rows["game"].append(ids[i])
                        rows["same_deeper"].append(deeper[i].move == res.move)
                        rows["same_expert"].append(mv_expert == res.move)
                        rows["fen"].append(b.fen())
                    prev_value[i][b.turn] = res.root.value
                    b.push(res.move)
                    if not b.is_game_over(claim_draw=True) and b.ply() < cc.max_plies:
                        nxt.append(i)
                active = nxt
            print(f"games {ids[-1] + 1}/{cc.n_dataset_games}, positions {len(rows['u'])}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        oracle.quit()
        expert.quit()

    out = cfg.run_dir / "controller_data.npz"
    np.savez(out, **{k: np.array(v) for k, v in rows.items()})
    print(f"Saved {len(rows['u'])} positions to {out}")


if __name__ == "__main__":
    generate(Config(), sys.argv[1])
