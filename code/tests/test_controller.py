import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

from config import Config
from controller.features import FEATURE_NAMES, extract_features
from controller.reward import action_rewards
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device


def test_features_shape():
    cfg = Config(); cfg.model.n_blocks, cfg.model.channels = 1, 16; cfg.device.device = "cpu"
    ev = Evaluator(ChessNet(cfg.model), get_device(cfg.device))
    b = chess.Board()
    res = run_batched(ev, [gumbel_search(b, 8, cfg.mcts, 0.0, np.random.default_rng(0))])[0]
    f = extract_features(b, res, None)
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


def _data(u):
    n = len(u)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32); feats[:, 0] = 0.5
    return {"features": feats, "u": np.array(u), "cp_base": np.full(n, 50.0),
            "cp_deeper": np.full(n, 80.0), "cp_expert": np.full(n, 50.0), "cp_random": np.full(n, -400.0)}


def test_rewards_no_fault():
    cc = Config().controller; cc.fault_mode = "none"
    r, g, fault = action_rewards(_data([0.0]), cc)
    assert not fault.any()
    assert np.allclose(g[0], [0, 30, 0, 0])
    assert np.allclose(r[0], [0, 30 - cc.cost_deeper, -cc.cost_expert, -cc.cost_judge])


def test_rewards_with_fault():
    cc = Config().controller; cc.fault_mode = "uniform"; cc.fault_rate = 0.5
    _, g, fault = action_rewards(_data([0.1, 0.9]), cc)
    assert list(fault) == [True, False]
    assert np.allclose(g[0], [0, 0, 450, 450])  # expert/judge avoid the -400 corrupted move


def test_tanh_transform_compresses_big_leads():
    cc = Config().controller; cc.fault_mode = "none"; cc.reward_transform = "cp_tanh"
    d = _data([0.0]); d["cp_base"][:] = 900; d["cp_deeper"][:] = 1200
    _, g, _ = action_rewards(d, cc)
    assert 0 < g[0, 1] < 300 * 0.5
