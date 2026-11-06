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


def test_considered_visits_matches_mctx_schedule():
    from mcts.gumbel import considered_visits
    # mctx get_sequence_of_considered_visits(16, 32): 16 actions x1, 8 x1, 4 x2; full budget used
    assert considered_visits(16, 32) == [0] * 16 + [1] * 8 + [2] * 4 + [3] * 4
    assert considered_visits(2, 6) == [0, 0, 1, 1, 2, 2]
    assert considered_visits(1, 3) == [0, 1, 2]
    for m, n in [(16, 8), (16, 64), (5, 32), (16, 200)]:
        assert len(considered_visits(m, n)) == n


def test_search_spends_full_budget_on_survivors():
    cfg = _tiny_cfg()
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    res = run_batched(ev, [gumbel_search(chess.Board(), 32, cfg.mcts, 1.0, np.random.default_rng(1))])[0]
    assert res.root.N.sum() == 32
    assert res.root.N[res.action] == res.root.N.max()   # chosen among the most-visited survivors


def test_q_normalization_modes():
    from mcts.gumbel import Node, sigma
    cfg = _tiny_cfg()
    node = Node([None] * 3, np.zeros(3), 0.0, None)
    q = np.array([-0.1, 0.0, 0.1])
    cfg.mcts.q_normalization = "minmax"
    assert np.allclose(sigma(node, q, cfg.mcts) / cfg.mcts.c_visit, [0.0, 0.5, 1.0])
    cfg.mcts.q_normalization = "fixed"
    assert np.allclose(sigma(node, q, cfg.mcts) / cfg.mcts.c_visit, [0.45, 0.5, 0.55])


def test_resign_calibration():
    from training.resign import ResignCalibrator, trigger_level
    assert trigger_level([0.0, -0.95, -0.96, -0.2, -0.97], 2) == -0.95
    assert trigger_level([-0.99], 2) is None
    cfg = Config().selfplay
    cfg.resign_min_samples = 10
    cal = ResignCalibrator(cfg)
    for _ in range(50):
        cal.add(-0.98, True)     # deep trouble: always lost
        cal.add(-0.85, False)    # only mildly bad: survived
    cal.update()
    assert -0.98 < cal.threshold <= -0.85   # resign below -0.85 only, false positives excluded
    assert cal.false_positive_rate(cal.threshold)[0] == 0.0


def test_resign_suspended_when_no_threshold_is_safe():
    from training.resign import RESIGN_OFF, ResignCalibrator
    cfg = Config().selfplay
    cfg.resign_min_samples, cfg.resign_window = 10, 100
    cal = ResignCalibrator(cfg)
    for _ in range(50):
        cal.add(-0.995, True)
        cal.add(-0.995, False)   # half of the would-be resignations at every threshold survive
    cal.update()
    assert cal.threshold == RESIGN_OFF
    for _ in range(100):         # value estimates become reliable: the old records leave the window
        cal.add(-0.98, True)
    cal.update()
    assert cfg.resign_bounds[0] <= cal.threshold <= cfg.resign_bounds[1]


def test_selfplay_resign_off_and_opening_exploration():
    cfg = _tiny_cfg()
    assert not cfg.selfplay.resign and cfg.selfplay.explore_plies == 0   # defaults
    cfg.selfplay.explore_plies = 1000      # sample every move by visit counts
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    _, stats = play_games(ev, cfg, 6, np.random.default_rng(0), resign_threshold=1.0)
    assert len(stats) == 6
    assert all(s["termination"] != "resign" and not s["resign_allowed"] for s in stats)


def test_paired_match_between_agents():
    from evaluate import AgentPlayer, play_match
    cfg = _tiny_cfg()
    cfg.eval.max_plies = 16
    a = AgentPlayer(cfg, ChessNet(cfg.model), 4, torch.device("cpu"))
    b = AgentPlayer(cfg, ChessNet(cfg.model), 4, torch.device("cpu"))
    w, d, l = play_match(a, b, cfg, np.random.default_rng(0), n_games=6, verbose=False)
    assert w + d + l == 6 and a.n_moves > 0 and b.n_moves > 0


def test_train_loop_resumes(tmp_path):
    import train_selfplay
    cfg = _tiny_cfg()
    cfg.paths.runs_dir = tmp_path
    cfg.selfplay.games_per_iteration = 4
    cfg.train.min_buffer_size, cfg.train.batch_size = 1, 16
    cfg.train.buffer_size, cfg.train.checkpoint_every, cfg.train.n_iterations = 500, 1, 2
    cfg.train_eval.every = 0
    train_selfplay.main(cfg)
    lines = (cfg.run_dir / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2 and (cfg.run_dir / "steps.jsonl").exists()
    cfg.train.n_iterations = 3
    train_selfplay.main(cfg)   # resumes from iteration 2
    assert len((cfg.run_dir / "log.jsonl").read_text().splitlines()) == 3
