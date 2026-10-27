"""Controller state features, computed from the base agent's own search output."""
import chess
import numpy as np

from mcts.gumbel import SearchResult, completed_q

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
FEATURE_NAMES = [
    "prior_entropy_norm", "prior_top", "improved_top", "improved_gap",
    "visit_top_share", "visit_gap", "q_gap", "p_win", "p_draw", "p_loss", "value",
    "value_delta", "log_legal", "material_balance", "total_material", "ply", "in_check",
]


def _top2(x):
    if len(x) < 2:
        return (x[0] if len(x) else 0.0), 0.0
    s = np.sort(x)
    return s[-1], s[-1] - s[-2]


def normalized_entropy(prior: np.ndarray) -> float:
    if len(prior) < 2:
        return 0.0
    p = prior[prior > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(prior)))


def extract_features(board: chess.Board, res: SearchResult, prev_value: float) -> np.ndarray:
    root = res.root
    h = normalized_entropy(root.prior)
    imp_top, imp_gap = _top2(res.policy_target)
    n_total = max(root.N.sum(), 1)
    v_top, v_gap = _top2(root.N / n_total)
    _, q_gap = _top2(completed_q(root))
    wdl = root.wdl
    mat = {c: sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == c)
           for c in (chess.WHITE, chess.BLACK)}
    us = board.turn
    return np.array([
        h, root.prior.max(), imp_top, imp_gap, v_top, v_gap, q_gap,
        wdl[0], wdl[1], wdl[2], root.value,
        0.0 if prev_value is None else root.value - prev_value,
        np.log(len(root.moves)), (mat[us] - mat[not us]) / 10.0, (mat[us] + mat[not us]) / 78.0,
        board.ply() / 100.0, float(board.is_check()),
    ], dtype=np.float32)
