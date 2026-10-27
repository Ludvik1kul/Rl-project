"""Batched self-play: many games stepped together, one Gumbel search per game per move."""
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_env.encoding import encode_board, encode_move, outcome_wdl
from config import Config
from mcts.gumbel import gumbel_search, run_batched
from training.replay_buffer import compress_planes, sparse_policy


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    allow_resign: bool = True
    low_value_streak: dict = field(default_factory=lambda: {chess.WHITE: 0, chess.BLACK: 0})
    records: list = field(default_factory=list)   # (planes_u8, pol_idx, pol_p, turn)
    result: str = None


def _game_result(gs: GameState, cfg: Config):
    b = gs.board
    if b.is_game_over(claim_draw=True):
        return b.result(claim_draw=True)
    if b.ply() >= cfg.selfplay.max_plies:
        return "1/2-1/2"
    return None


def play_games(evaluator, cfg: Config, n_games: int, rng: np.random.Generator):
    """Plays n_games of self-play. Returns (training samples, list of game stats)."""
    sp, mc = cfg.selfplay, cfg.mcts
    active, finished = [], []
    started = 0

    def new_game():
        return GameState(allow_resign=rng.random() >= sp.no_resign_fraction)

    while started < min(n_games, sp.n_parallel_games):
        active.append(new_game())
        started += 1

    while active:
        full = rng.random(len(active)) < sp.full_search_prob
        gens = []
        for gs, is_full in zip(active, full):
            scale = mc.gumbel_scale if gs.board.ply() < sp.gumbel_plies else 0.0
            sims = mc.n_simulations if is_full else sp.fast_simulations
            gens.append(gumbel_search(gs.board, sims, mc, scale, rng))
        results = run_batched(evaluator, gens)

        still_active = []
        for gs, is_full, res in zip(active, full, results):
            turn = gs.board.turn
            if is_full:
                idx = np.array([encode_move(m, turn) for m in res.policy_moves], dtype=np.int64)
                pi, pp = sparse_policy(idx, res.policy_target)
                gs.records.append((compress_planes(encode_board(gs.board)), pi, pp, turn))
            # resignation
            if res.root.value < sp.resign_threshold:
                gs.low_value_streak[turn] += 1
            else:
                gs.low_value_streak[turn] = 0
            if gs.allow_resign and gs.low_value_streak[turn] >= sp.resign_consecutive:
                gs.result = "0-1" if turn == chess.WHITE else "1-0"
            else:
                gs.board.push(res.move)
                gs.result = _game_result(gs, cfg)
            if gs.result is None:
                still_active.append(gs)
            else:
                finished.append(gs)
                if started < n_games:
                    still_active.append(new_game())
                    started += 1
        active = still_active

    planes, pol_idx, pol_p, wdl, stats = [], [], [], [], []
    for gs in finished:
        for pl, pi, pp, turn in gs.records:
            planes.append(pl); pol_idx.append(pi); pol_p.append(pp)
            wdl.append(outcome_wdl(gs.result, turn))
        stats.append({"result": gs.result, "plies": gs.board.ply(), "resign_allowed": gs.allow_resign})
    if not planes:
        return None, stats
    samples = (np.stack(planes), np.stack(pol_idx), np.stack(pol_p), np.array(wdl, dtype=np.int8))
    return samples, stats
