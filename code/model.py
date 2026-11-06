"""Residual policy / WDL-value network and device helpers."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, DeviceConfig


def get_device(cfg: DeviceConfig) -> torch.device:
    torch.set_num_threads(cfg.torch_threads)
    if cfg.device != "auto":
        return torch.device(cfg.device)
    candidates = []
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        candidates.append("xpu")
    if torch.cuda.is_available():
        candidates.append("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        candidates.append("mps")
    for name in candidates:
        try:  # smoke test: a driver problem should fall back, not crash a multi-day run
            dev = torch.device(name)
            conv = nn.Conv2d(2, 2, 3, padding=1).to(dev)
            conv(torch.ones(1, 2, 8, 8, device=dev)).sum().backward()
            return dev
        except Exception as e:
            print(f"Device {name} failed smoke test ({e}); trying next")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    """
    Outputs policy logits (n_moves) and WDL logits (3) for the side to move.

    Based on Alphazero architecture (Silver et al. 2018), scaled down. 
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(c) for _ in range(cfg.n_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, cfg.policy_channels, 1, bias=False), nn.BatchNorm2d(cfg.policy_channels),
            nn.ReLU(), nn.Flatten(), nn.Linear(cfg.policy_channels * 64, cfg.n_moves))
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * 64, cfg.value_hidden), nn.ReLU(), nn.Linear(cfg.value_hidden, 3))

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.policy_head(h), self.value_head(h)


class Evaluator:
    """Batched inference wrapper: planes -> (policy logits, WDL probs) as numpy."""

    def __init__(self, net: ChessNet, device: torch.device):
        self.net = net.to(device).eval()
        self.device = device
        self.n_calls = self.n_positions = 0   # throughput counters

    @torch.inference_mode()
    def __call__(self, planes: np.ndarray):
        # The same net is trained between self-play rounds (train_step sets train mode);
        # BatchNorm must use running stats here, not per-batch stats.
        self.net.eval()
        self.n_calls += 1
        self.n_positions += len(planes)
        x = torch.from_numpy(planes).to(self.device)
        logits, wdl = self.net(x)
        return logits.float().cpu().numpy(), torch.softmax(wdl.float(), -1).cpu().numpy()


def save_checkpoint(path, net, optimizer=None, **extra):
    state = {"model": net.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, net, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    net.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
