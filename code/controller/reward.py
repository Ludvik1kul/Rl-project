"""Reward and fault model, computed offline from stored oracle evaluations.

The dataset stores raw oracle centipawn scores (mover's perspective) of the
position after each candidate move, plus a uniform draw `u` per position, so the
reward transform, costs and fault mode can all be changed without regenerating data.

Per-action reward = T(eval after action's move) - T(eval after the play action's move) - cost.
"""
import numpy as np

from config import ControllerConfig

ACTIONS = ["play", "deeper", "expert", "judge"]


def transform(cp: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    cp = np.asarray(cp, dtype=np.float64)
    if cfg.reward_transform == "cp":
        return np.clip(cp, -cfg.cp_clip, cfg.cp_clip)
    if cfg.reward_transform == "cp_tanh":
        return cfg.cp_scale * np.tanh(cp / cfg.cp_scale)
    if cfg.reward_transform == "expected_score":
        return 1000.0 / (1.0 + np.exp(-0.00368208 * cp))
    raise ValueError(cfg.reward_transform)


def fault_probability(entropy_norm: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    if cfg.fault_mode == "none":
        return np.zeros_like(entropy_norm)
    if cfg.fault_mode == "uniform":
        return np.full_like(entropy_norm, cfg.fault_rate)
    if cfg.fault_mode == "uncertainty":
        return np.minimum(1.0, cfg.fault_rate * (entropy_norm / 0.5) ** cfg.fault_uncertainty_power)
    raise ValueError(cfg.fault_mode)


def costs(cfg: ControllerConfig) -> np.ndarray:
    return np.array([cfg.cost_play, cfg.cost_deeper, cfg.cost_expert, cfg.cost_judge])


def action_rewards(data: dict, cfg: ControllerConfig):
    """Returns (rewards (N,4), gains (N,4) without costs, fault flags (N,))."""
    fault = data["u"] < fault_probability(data["features"][:, 0], cfg)
    t = {k: transform(data[k], cfg) for k in ("cp_base", "cp_deeper", "cp_expert", "cp_random")}
    outcome = np.stack([
        np.where(fault, t["cp_random"], t["cp_base"]),    # play: agent move, possibly corrupted
        np.where(fault, t["cp_random"], t["cp_deeper"]),  # deeper: agent move, possibly corrupted
        t["cp_expert"],                                   # expert bypasses the agent
        t["cp_base"],                                     # judge catches and corrects any fault
    ], axis=1)
    gains = outcome - outcome[:, :1]
    return gains - costs(cfg), gains, fault
