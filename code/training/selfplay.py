"""Batched self-play: many games stepped together, one Gumbel search per game per move."""
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_env.encoding import encode_board, encode_move, outcome_wdl
from config import Config
from mcts.gumbel import gumbel_search
from training.replay_buffer import compress_planes, sparse_policy


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    allow_resign: bool = True
    low_value_streak: dict = field(default_factory=lambda: {chess.WHITE: 0, chess.BLACK: 0})
    records: list = field(default_factory=list)   # (planes_u8, pol_idx, pol_p, turn, search value)
    values: dict = field(default_factory=lambda: {chess.WHITE: [], chess.BLACK: []})  # search values
    result: str = None
    termination: str = None


def _game_result(gs: GameState, cfg: Config):
    b = gs.board
    outcome = b.outcome(claim_draw=True)
    if outcome is not None:
        gs.termination = outcome.termination.name.lower()
        return outcome.result()
    if b.ply() >= cfg.selfplay.max_plies:
        gs.termination = "max_plies"
        return "1/2-1/2"
    return None


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def play_games(evaluator, cfg: Config, n_games: int, rng: np.random.Generator, resign_threshold=None,
               resign_enabled: bool = True):
    """Plays n_games of self-play. Returns (training samples, list of game stats).

    Resignation uses the root search value and `resign_threshold` (default: the configured one).
    With resign_enabled=False every game is played out (and counts as a no-resign game).
    """
    sp, mc = cfg.selfplay, cfg.mcts
    threshold = sp.resign_threshold if resign_threshold is None else resign_threshold
    finished = []
    started = 0

    def new_game():
        return GameState(allow_resign=resign_enabled and rng.random() >= sp.no_resign_fraction)

    def apply_result(gs: GameState, is_full: bool, res) -> bool:
        """Records the search, checks resignation, plays the move. True if the game is over."""
        turn = gs.board.turn
        value = res.value
        gs.values[turn].append(value)
        if is_full:
            idx = np.array([encode_move(m, turn) for m in res.policy_moves], dtype=np.int64)
            pi, pp = sparse_policy(idx, res.policy_target)
            gs.records.append((compress_planes(encode_board(gs.board)), pi, pp, turn, value,
                               _entropy(res.policy_target),
                               float((res.policy_target * (np.log(res.policy_target + 1e-12)
                                                           - np.log(res.root.prior + 1e-12))).sum())))
        if value < threshold:
            gs.low_value_streak[turn] += 1
        else:
            gs.low_value_streak[turn] = 0
        if gs.allow_resign and gs.low_value_streak[turn] >= sp.resign_consecutive:
            gs.result = "0-1" if turn == chess.WHITE else "1-0"
            gs.termination = "resign"
        else:
            gs.board.push(res.move)
            gs.result = _game_result(gs, cfg)
        return gs.result is not None

    def launch(gs: GameState):
        """Starts the next search of gs: a slot [game, search, is_full, pending leaf], or None if the game ended."""
        while True:
            is_full = rng.random() < sp.full_search_prob
            scale = mc.gumbel_scale if gs.board.ply() < sp.gumbel_plies else 0.0
            sims = mc.n_simulations if is_full else sp.fast_simulations
            gen = gumbel_search(gs.board, sims, mc, scale, rng)
            try:
                return [gs, gen, is_full, next(gen)]
            except StopIteration as stop:   # search needed no network call
                if apply_result(gs, is_full, stop.value):
                    return None

    def fill(slots):
        """Launches games until n_parallel_games are running or n_games have been started."""
        nonlocal started
        while started < n_games and len(slots) < sp.n_parallel_games:
            gs = new_game()
            started += 1
            slot = launch(gs)
            if slot is None:
                finished.append(gs)
            else:
                slots.append(slot)

    # Each game advances on its own: when its search finishes it plays the move and starts the
    # next search at once, so the batch stays full while other games are still in longer searches.
    slots = []
    fill(slots)
    while slots:
        logits, wdl = evaluator(np.stack([s[3] for s in slots]))
        still = []
        for k, slot in enumerate(slots):
            gs, gen, is_full, _ = slot
            try:
                slot[3] = gen.send((logits[k], wdl[k]))
                still.append(slot)
                continue
            except StopIteration as stop:
                res = stop.value
            nxt = None if apply_result(gs, is_full, res) else launch(gs)
            if nxt is None:
                finished.append(gs)
            else:
                still.append(nxt)
        slots = still
        fill(slots)

    planes, pol_idx, pol_p, wdl, stats = [], [], [], [], []
    for gs in finished:
        for pl, pi, pp, turn, *_ in gs.records:
            planes.append(pl); pol_idx.append(pi); pol_p.append(pp)
            wdl.append(outcome_wdl(gs.result, turn))
        z = {chess.WHITE: {"1-0": 1, "0-1": -1}.get(gs.result, 0)}
        z[chess.BLACK] = -z[chess.WHITE]
        stats.append({
            "result": gs.result, "plies": gs.board.ply(), "termination": gs.termination,
            "resign_allowed": gs.allow_resign,
            "lost": {side: z[side] < 0 for side in z},
            "values": gs.values,
            "positions": len(gs.records),
            # per full search: (policy target entropy, KL(target || prior), (search value - z)^2)
            "search": [(e, kl, (v - z[t]) ** 2) for *_, t, v, e, kl in gs.records],
        })
    if not planes:
        return None, stats
    samples = (np.stack(planes), np.stack(pol_idx), np.stack(pol_p), np.array(wdl, dtype=np.int8))
    return samples, stats
