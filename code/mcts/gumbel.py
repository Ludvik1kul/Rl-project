"""Gumbel AlphaZero search (Danihelka et al., 2022, "Policy improvement by planning with Gumbel").

Root: Gumbel-top-k sampling of m candidate actions + sequential halving.
Interior nodes: deterministic selection argmax(pi'(a) - N(a) / (1 + sum N)),
where pi' = softmax(logits + sigma(completed Q)).

A search is a generator: it yields encoded leaf planes and expects
(policy_logits, wdl_probs) to be sent back. `run_batched` drives many such
generators at once so all leaves are evaluated in a single batch.
All values are in [-1, 1] from the perspective of the side to move at the node.
"""
import math
from dataclasses import dataclass

import chess
import numpy as np

from chess_env.encoding import encode_board, legal_move_indices
from config import MCTSConfig


class Node:
    __slots__ = ("moves", "logits", "prior", "N", "W", "children", "value", "wdl", "terminal")

    def __init__(self, moves, logits, value, wdl, terminal=False):
        self.moves = moves
        self.logits = logits                     # raw logits over legal moves
        if len(moves):
            p = np.exp(logits - logits.max())
            self.prior = p / p.sum()
        else:
            self.prior = logits
        self.N = np.zeros(len(moves), dtype=np.int32)
        self.W = np.zeros(len(moves), dtype=np.float64)
        self.children = [None] * len(moves)
        self.value = value                       # network (or terminal) value estimate
        self.wdl = wdl
        self.terminal = terminal


@dataclass
class SearchResult:
    move: chess.Move
    action: int                  # index into root.moves
    policy_moves: list           # legal moves at root
    policy_target: np.ndarray    # improved policy over policy_moves
    root: Node


def _terminal_value(board: chess.Board):
    """Value for side to move if the game is over, else None."""
    if board.is_checkmate():
        return -1.0
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_fifty_moves() or board.is_repetition(3)):
        return 0.0
    return None


def _expand(board: chess.Board):
    """Generator: creates a node for the current board, querying the network if needed."""
    tv = _terminal_value(board)
    if tv is not None:
        return Node([], np.zeros(0), tv, None, terminal=True)
    moves, idx = legal_move_indices(board)
    logits, wdl = yield encode_board(board)
    return Node(moves, logits[idx].astype(np.float64), float(wdl[0] - wdl[2]), wdl)


def completed_q(node: Node) -> np.ndarray:
    """Q per action from the node's perspective, unvisited actions filled with v_mix."""
    visited = node.N > 0
    q = np.zeros(len(node.moves))
    q[visited] = node.W[visited] / node.N[visited]
    sum_n = node.N.sum()
    if sum_n == 0:
        v_mix = node.value
    else:
        pv = node.prior[visited]
        weighted = (pv * q[visited]).sum() / max(pv.sum(), 1e-12)
        v_mix = (node.value + sum_n * weighted) / (1 + sum_n)
    q[~visited] = v_mix
    return q


def sigma(node: Node, q: np.ndarray, cfg: MCTSConfig) -> np.ndarray:
    q01 = (q + 1.0) / 2.0
    return (cfg.c_visit + node.N.max(initial=0)) * cfg.c_scale * q01


def improved_policy(node: Node, cfg: MCTSConfig) -> np.ndarray:
    z = node.logits + sigma(node, completed_q(node), cfg)
    p = np.exp(z - z.max())
    return p / p.sum()


def _simulate(board: chess.Board, root: Node, root_action: int, cfg: MCTSConfig):
    """Generator: one simulation through `root_action`, with backup."""
    path = []
    node, a = root, root_action
    while True:
        path.append((node, a))
        board.push(node.moves[a])
        child = node.children[a]
        if child is None:
            child = yield from _expand(board)
            node.children[a] = child
            v = child.value
            break
        if child.terminal:
            v = child.value
            break
        pi = improved_policy(child, cfg)
        a = int(np.argmax(pi - child.N / (1.0 + child.N.sum())))
        node = child
    for parent, act in reversed(path):
        v = -v
        parent.N[act] += 1
        parent.W[act] += v
        board.pop()


def gumbel_search(board: chess.Board, n_simulations: int, cfg: MCTSConfig,
                  gumbel_scale: float, rng: np.random.Generator):
    """Generator returning a SearchResult. Board is restored on completion."""
    root = yield from _expand(board)
    n = len(root.moves)
    g = gumbel_scale * rng.gumbel(size=n)
    m = min(cfg.max_considered_actions, n, n_simulations)
    remaining = list(np.argsort(-(g + root.logits))[:m])
    if n > 1:
        n_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        used = 0
        for _ in range(n_phases):
            per_action = max(1, n_simulations // (n_phases * len(remaining)))
            for _ in range(per_action):
                for a in remaining:
                    if used >= n_simulations:
                        break
                    yield from _simulate(board, root, int(a), cfg)
                    used += 1
            score = g + root.logits + sigma(root, completed_q(root), cfg)
            remaining.sort(key=lambda a: -score[a])
            remaining = remaining[:max(1, len(remaining) // 2)]
    action = int(remaining[0])
    return SearchResult(root.moves[action], action, root.moves, improved_policy(root, cfg), root)


def run_batched(evaluator, generators):
    """Drive several search generators, batching their leaf evaluations."""
    results = [None] * len(generators)
    requests = {}
    for i, gen in enumerate(generators):
        try:
            requests[i] = next(gen)
        except StopIteration as stop:
            results[i] = stop.value
    while requests:
        ids = list(requests)
        logits, wdl = evaluator(np.stack([requests[i] for i in ids]))
        nxt = {}
        for k, i in enumerate(ids):
            try:
                nxt[i] = generators[i].send((logits[k], wdl[k]))
            except StopIteration as stop:
                results[i] = stop.value
        requests = nxt
    return results
