"""Rating assessment: matches against Stockfish (limited strength) or another checkpoint.

Usage:
    python evaluate.py stockfish runs/main/latest.pt
    python evaluate.py match runs/main/checkpoints/iter_00100.pt runs/main/checkpoints/iter_00010.pt
"""
import math
import sys

import chess
import chess.engine
import numpy as np

from config import Config
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device, load_checkpoint


class AgentPlayer:
    def __init__(self, cfg: Config, ckpt_path, sims: int):
        device = get_device(cfg.device)
        net = ChessNet(cfg.model)
        load_checkpoint(ckpt_path, net, map_location=device)
        self.evaluator = Evaluator(net, device)
        self.cfg, self.sims = cfg, sims
        self.rng = np.random.default_rng(0)

    def play(self, board: chess.Board) -> chess.Move:
        gen = gumbel_search(board, self.sims, self.cfg.mcts, 0.0, self.rng)
        return run_batched(self.evaluator, [gen])[0].move


class StockfishPlayer:
    def __init__(self, cfg: Config):
        ec = cfg.eval
        self.engine = chess.engine.SimpleEngine.popen_uci(str(cfg.paths.stockfish_path))
        if ec.opponent_skill_level >= 0:
            self.engine.configure({"Skill Level": ec.opponent_skill_level})
        else:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": ec.opponent_elo})
        self.limit = chess.engine.Limit(time=ec.opponent_time)

    def play(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


def play_match(p1, p2, cfg: Config, rng: np.random.Generator):
    """Plays cfg.eval.n_games alternating colours. Returns (wins, draws, losses) for p1."""
    ec = cfg.eval
    w = d = l = 0
    for g in range(ec.n_games):
        board = chess.Board()
        for _ in range(ec.random_opening_plies):
            board.push(rng.choice(list(board.legal_moves)))
        white, black = (p1, p2) if g % 2 == 0 else (p2, p1)
        while not board.is_game_over(claim_draw=True) and board.ply() < ec.max_plies:
            board.push((white if board.turn == chess.WHITE else black).play(board))
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        p1_white = g % 2 == 0
        if result == "1/2-1/2":
            d += 1
        elif (result == "1-0") == p1_white:
            w += 1
        else:
            l += 1
        print(f"game {g + 1}: {result} (p1 {'white' if p1_white else 'black'})  W/D/L {w}/{d}/{l}")
    return w, d, l


def elo_estimate(w, d, l, z=1.96):
    """Elo difference with a normal-approximation confidence interval."""
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    to_elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    return to_elo(s), to_elo(s - z * se), to_elo(s + z * se)


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
