"""Run with: python -m pytest tests (from the code/ folder)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import torch

from chess_env.encoding import N_MOVES, decode_move, encode_board, encode_move
from config import Config
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device
from training.losses import train_step
from training.selfplay import play_games


def test_move_roundtrip_random_games():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        b = chess.Board()
        while not b.is_game_over() and b.ply() < 200:
            for m in b.legal_moves:
                i = encode_move(m, b.turn)
                assert 0 <= i < N_MOVES
                assert decode_move(i, b) == m
                seen.add((b.fen(), i))
            b.push(rng.choice(list(b.legal_moves)))


def test_underpromotion_roundtrip():
    for fen in ["8/1P6/8/8/8/8/k7/4K3 w - - 0 1", "4k3/8/8/8/8/8/1p6/K7 b - - 0 1"]:
        b = chess.Board(fen)
        idx = {encode_move(m, b.turn) for m in b.legal_moves}
        assert len(idx) == b.legal_moves.count()
        for m in b.legal_moves:
            assert decode_move(encode_move(m, b.turn), b) == m


def test_encoding_symmetry():
    # Initial position looks identical for both sides after flipping.
    b = chess.Board()
    w = encode_board(b)
    b.push_san("Nf3"); b.push_san("Nf6"); b.push_san("Ng1"); b.push_san("Ng8")
    assert np.allclose(encode_board(b)[:18], w[:18])


def _tiny_cfg():
    cfg = Config()
    cfg.model.n_blocks, cfg.model.channels = 1, 16
    cfg.device.device = "cpu"
    cfg.mcts.n_simulations = 8
    cfg.selfplay.n_parallel_games = 4
    cfg.selfplay.max_plies = 20
    return cfg


def test_search_returns_legal_move_and_distribution():
    cfg = _tiny_cfg()
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    b = chess.Board()
    fen = b.fen()
    res = run_batched(ev, [gumbel_search(b, 16, cfg.mcts, 1.0, np.random.default_rng(0))])[0]
    assert b.fen() == fen  # board restored
    assert res.move in b.legal_moves
    assert abs(res.policy_target.sum() - 1) < 1e-6
    assert res.root.N.sum() == 16


def test_mate_in_one_found():
    cfg = _tiny_cfg()
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")  # Ra8#
    cfg.mcts.max_considered_actions = 32
    res = run_batched(ev, [gumbel_search(b, 200, cfg.mcts, 0.0, np.random.default_rng(0))])[0]
    assert res.move == chess.Move.from_uci("a1a8")


def test_evaluator_uses_eval_mode_after_training():
    # Self-play shares the net with the trainer; inference must not depend on batch
    # composition or update BatchNorm running stats after a train_step.
    cfg = _tiny_cfg()
    dev = get_device(cfg.device)
    net = ChessNet(cfg.model)
    ev = Evaluator(net, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x = np.random.default_rng(0).random((8, 20, 8, 8)).astype(np.float32)
    train_step(net, opt, x, np.ones(8, dtype=np.int64), np.zeros(8), dev)
    stats = [b.clone() for b in net.buffers()]
    single, batch = ev(x[:1]), ev(x)
    assert np.allclose(single[0], batch[0][:1], atol=1e-5)
    assert np.allclose(single[1], batch[1][:1], atol=1e-5)
    assert all(torch.equal(a, b) for a, b in zip(stats, net.buffers()))


def test_selfplay_smoke():
    cfg = _tiny_cfg()
    cfg.selfplay.full_search_prob = 1.0
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    samples, stats = play_games(ev, cfg, 6, np.random.default_rng(0))
    assert len(stats) == 6
    planes, pi, pp, wdl = samples
    assert planes.shape[1:] == (20, 8, 8) and len(planes) == len(wdl)
