"""Contextual-bandit controller trained on the counterfactual dataset, plus baselines.

The MLP predicts the cost-free gain of every action; the policy is
argmax_a (predicted_gain_a - lambda * cost_a). Sweeping lambda traces a
gain-vs-cost curve that is compared with threshold heuristics (Pareto plot).

Usage:
    python -m controller.bandit            # uses runs/<run_name>/controller_data.npz
"""
import json

import numpy as np
import torch
import torch.nn as nn

from config import Config
from controller.features import FEATURE_NAMES
from controller.reward import ACTIONS, action_rewards, costs

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


class GainModel(nn.Module):
    def __init__(self, n_in, hidden, n_actions=len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def split_by_game(game, val_fraction, rng):
    games = np.unique(game)
    val_games = rng.choice(games, size=max(1, int(len(games) * val_fraction)), replace=False)
    val = np.isin(game, val_games)
    return ~val, val


def train_model(x, gains, cfg):
    cc = cfg.controller
    torch.manual_seed(cfg.train.seed)
    model = GainModel(x.shape[1], cc.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)
    xt, yt = torch.tensor(x), torch.tensor(gains, dtype=torch.float32)
    scale = yt.std() + 1e-6  # regress on a normalised scale
    for _ in range(cc.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.smooth_l1_loss(model(xt[b]), yt[b] / scale)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return lambda z: model(torch.tensor(z)).detach().numpy() * scale.item()


def summarize(choice, gains, cost_vec):
    n = len(choice)
    idx = np.arange(n)
    return {"gain": float(gains[idx, choice].mean()), "cost": float(cost_vec[choice].mean()),
            "net": float((gains[idx, choice] - cost_vec[choice]).mean()),
            "freq": {a: float((choice == k).mean()) for k, a in enumerate(ACTIONS)}}


def evaluate_policies(x_val, gains_val, predict, cfg):
    c = costs(cfg.controller)
    n = len(x_val)
    res = {f"always_{a}": summarize(np.full(n, k), gains_val, c) for k, a in enumerate(ACTIONS)}
    res["oracle"] = summarize(np.argmax(gains_val - c, axis=1), gains_val, c)
    pred = predict(x_val)
    res["learned"] = {str(l): summarize(np.argmax(pred - l * c, axis=1), gains_val, c) for l in LAMBDAS}
    # Heuristic: consult expert when the top move's visit share is below a threshold
    vt = x_val[:, FEATURE_NAMES.index("visit_top_share")]
    ent = x_val[:, FEATURE_NAMES.index("prior_entropy_norm")]
    res["threshold_expert"] = {
        f"{t:.2f}": summarize(np.where(vt < t, 2, 0), gains_val, c) for t in np.linspace(0, 1, 11)}
    res["threshold_judge"] = {
        f"{t:.2f}": summarize(np.where(ent > t, 3, 0), gains_val, c) for t in np.linspace(0, 1, 11)}
    return res


def plot_pareto(res, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, marker in [("learned", "o-"), ("threshold_expert", "s--"), ("threshold_judge", "^--")]:
        pts = sorted((v["cost"], v["gain"]) for v in res[name].values())
        ax.plot(*zip(*pts), marker, label=name, markersize=4)
    for a in ACTIONS:
        r = res[f"always_{a}"]
        ax.scatter(r["cost"], r["gain"], marker="x", label=f"always {a}")
    ax.scatter(res["oracle"]["cost"], res["oracle"]["gain"], marker="*", s=120, label="oracle")
    ax.set_xlabel("mean cost per move"); ax.set_ylabel("mean gain per move (reward units)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main(cfg: Config):
    data = dict(np.load(cfg.run_dir / "controller_data.npz"))
    rng = np.random.default_rng(cfg.train.seed)
    _, gains, fault = action_rewards(data, cfg.controller)
    x = data["features"].astype(np.float32)
    tr, val = split_by_game(data["game"], cfg.controller.val_fraction, rng)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    predict_std = train_model((x[tr] - mu) / sd, gains[tr], cfg)
    res = evaluate_policies(x[val], gains[val], lambda z: predict_std((z - mu) / sd), cfg)
    res["meta"] = {"n_train": int(tr.sum()), "n_val": int(val.sum()), "fault_rate": float(fault.mean()),
                   "same_deeper": float(data["same_deeper"].mean()),
                   "same_expert": float(data["same_expert"].mean()),
                   "config": {k: str(v) for k, v in vars(cfg.controller).items()}}
    tag = f"{cfg.controller.fault_mode}_{cfg.controller.reward_transform}"
    out = cfg.run_dir / f"controller_results_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved {out}")
    for name in ("always_play", "always_deeper", "always_expert", "always_judge", "oracle"):
        print(f"{name:15s} {res[name]}")
    print(f"{'learned l=1':15s} {res['learned']['1.0']}")
    plot_pareto(res, cfg.run_dir / f"controller_pareto_{tag}.png")


if __name__ == "__main__":
    main(Config())
