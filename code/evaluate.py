"""Rating assessment: matches against Stockfish (limited strength) or another checkpoint.

Games run in parallel so the agent's searches share one network batch. Each random opening is
played twice with colours swapped (paired games reduce variance).

Usage:
    python evaluate.py stockfish runs/main/latest.pt
    python evaluate.py match runs/main/checkpoints/iter_00100.pt runs/main/checkpoints/iter_00010.pt
"""
import math
import sys
import time

import chess
import chess.engine
import numpy as np

from chess_env.encoding import encode_board, legal_move_indices
from config import Config
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device, load_checkpoint


class AgentPlayer:
    """Deterministic Gumbel search (no noise). Accepts a net or a checkpoint path."""

    def __init__(self, cfg: Config, net_or_path, sims: int, device=None):
        device = device or get_device(cfg.device)
        if isinstance(net_or_path, ChessNet):
            net = net_or_path
        else:
            net = ChessNet(cfg.model)
            load_checkpoint(net_or_path, net, map_location=device)
        self.evaluator = Evaluator(net, device)
        self.cfg, self.sims = cfg, sims
        self.rng = np.random.default_rng(0)
        self.search_time, self.n_moves = 0.0, 0

    def play_batch(self, boards):
        t0 = time.perf_counter()
        gens = [gumbel_search(b, self.sims, self.cfg.mcts, 0.0, self.rng) for b in boards]
        moves = [r.move for r in run_batched(self.evaluator, gens)]
        self.search_time += time.perf_counter() - t0
        self.n_moves += len(boards)
        return moves

    def play(self, board: chess.Board) -> chess.Move:
        return self.play_batch([board])[0]

    def policy_move(self, board: chess.Board) -> chess.Move:
        """Most probable move of the raw policy (no search)."""
        moves, idx = legal_move_indices(board)
        logits, _ = self.evaluator(encode_board(board)[None])
        return moves[int(np.argmax(logits[0][idx]))]


class StockfishPlayer:
    def __init__(self, cfg: Config, elo: int = None, move_time: float = None):
        ec = cfg.eval
        self.engine = chess.engine.SimpleEngine.popen_uci(str(cfg.paths.stockfish_path))
        if ec.opponent_skill_level >= 0 and elo is None:
            self.engine.configure({"Skill Level": ec.opponent_skill_level})
        else:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo or ec.opponent_elo})
        self.limit = chess.engine.Limit(time=move_time if move_time is not None else ec.opponent_time)
        self.search_time, self.n_moves = 0.0, 0

    def play_batch(self, boards):
        t0 = time.perf_counter()
        moves = [self.engine.play(b, self.limit).move for b in boards]
        self.search_time += time.perf_counter() - t0
        self.n_moves += len(boards)
        return moves

    def play(self, board):
        return self.play_batch([board])[0]

    def close(self):
        self.engine.quit()


def random_opening(n_plies: int, rng: np.random.Generator) -> chess.Board:
    while True:
        board = chess.Board()
        for _ in range(n_plies):
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            return board


def _finished(board: chess.Board, max_plies: int) -> bool:
    return board.ply() >= max_plies or board.is_game_over(claim_draw=True)


def play_match(p1, p2, cfg: Config, rng: np.random.Generator, n_games: int = None, verbose: bool = True):
    """Plays n_games (default cfg.eval.n_games) as colour-swapped pairs. Returns (wins, draws, losses) for p1."""
    ec = cfg.eval
    n_games = n_games or ec.n_games
    openings = [random_opening(ec.random_opening_plies, rng) for _ in range((n_games + 1) // 2)]
    pending = [(g, openings[g // 2].copy(), g % 2 == 0) for g in range(n_games)]  # (id, board, p1 white)
    active, w, d, l = [], 0, 0, 0
    while pending or active:
        while pending and len(active) < ec.max_parallel_games:
            active.append(pending.pop(0))
        for is_p1, player in ((True, p1), (False, p2)):
            to_move = [b for _, b, p1_white in active
                       if ((b.turn == chess.WHITE) == p1_white) == is_p1 and not _finished(b, ec.max_plies)]
            if to_move:
                for b, m in zip(to_move, player.play_batch(to_move)):
                    b.push(m)
        still = []
        for g, b, p1_white in active:
            if not _finished(b, ec.max_plies):
                still.append((g, b, p1_white))
                continue
            result = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "1/2-1/2"
            if result == "1/2-1/2":
                d += 1
            elif (result == "1-0") == p1_white:
                w += 1
            else:
                l += 1
            if verbose:
                print(f"game {g + 1}: {result} (p1 {'white' if p1_white else 'black'})  W/D/L {w}/{d}/{l}")
        active = still
    return w, d, l


def elo_estimate(w, d, l, z=1.96):
    """Elo difference with a normal-approximation confidence interval."""
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    to_elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    return to_elo(s), to_elo(s - z * se), to_elo(s + z * se)


def time_moves(agent: AgentPlayer, boards, sims: int):
    """Seconds per move with batch size 1: raw policy (one forward pass) and full search."""
    t0 = time.perf_counter()
    for b in boards:
        agent.policy_move(b)
    t_policy = (time.perf_counter() - t0) / len(boards)
    t0 = time.perf_counter()
    for b in boards:
        run_batched(agent.evaluator, [gumbel_search(b, sims, agent.cfg.mcts, 0.0, agent.rng)])
    return t_policy, (time.perf_counter() - t0) / len(boards)


if __name__ == "__main__":
    cfg = Config()
    rng = np.random.default_rng(1)
    agent = AgentPlayer(cfg, sys.argv[2], cfg.eval.sims)
    if sys.argv[1] == "stockfish":
        opp = StockfishPlayer(cfg)
        try:
            wdl = play_match(agent, opp, cfg, rng)
        finally:
            opp.close()
        base = cfg.eval.opponent_elo
    else:
        wdl = play_match(agent, AgentPlayer(cfg, sys.argv[3], cfg.eval.sims), cfg, rng)
        base = 0
    diff, lo, hi = elo_estimate(*wdl)
    print(f"W/D/L {wdl}: Elo diff {diff:+.0f} [{lo:+.0f}, {hi:+.0f}] -> estimate {base + diff:.0f}")
    print(f"agent: {agent.search_time / max(agent.n_moves, 1):.3f} s/move (batched over games)")
