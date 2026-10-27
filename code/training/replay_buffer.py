"""Fixed-size ring buffer of training positions, stored compactly in numpy.

Planes are stored as uint8 (plane 19, the halfmove clock, is stored as clock*100).
Policy targets are sparse: up to MAX_POLICY move indices + probabilities.
"""
from pathlib import Path

import numpy as np

from chess_env.encoding import N_PLANES

MAX_POLICY = 128
HALFMOVE_PLANE = 19


def compress_planes(planes: np.ndarray) -> np.ndarray:
    out = planes.copy()
    out[..., HALFMOVE_PLANE, :, :] *= 100.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def decompress_planes(planes_u8: np.ndarray) -> np.ndarray:
    out = planes_u8.astype(np.float32)
    out[..., HALFMOVE_PLANE, :, :] /= 100.0
    return out


def sparse_policy(indices: np.ndarray, probs: np.ndarray):
    """Truncate to the MAX_POLICY most likely moves, renormalise, pad with -1."""
    order = np.argsort(-probs)[:MAX_POLICY]
    idx = np.full(MAX_POLICY, -1, dtype=np.int16)
    p = np.zeros(MAX_POLICY, dtype=np.float16)
    sel = probs[order]
    idx[:len(order)] = indices[order]
    p[:len(order)] = sel / sel.sum()
    return idx, p


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.planes = np.zeros((capacity, N_PLANES, 8, 8), dtype=np.uint8)
        self.pol_idx = np.full((capacity, MAX_POLICY), -1, dtype=np.int16)
        self.pol_p = np.zeros((capacity, MAX_POLICY), dtype=np.float16)
        self.wdl = np.zeros(capacity, dtype=np.int8)
        self.size = 0
        self.pos = 0
        self.total_added = 0

    def add(self, planes_u8, pol_idx, pol_p, wdl):
        for i in range(len(wdl)):
            self.planes[self.pos] = planes_u8[i]
            self.pol_idx[self.pos] = pol_idx[i]
            self.pol_p[self.pos] = pol_p[i]
            self.wdl[self.pos] = wdl[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self.total_added += len(wdl)

    def sample(self, batch_size: int, rng: np.random.Generator):
        ix = rng.integers(0, self.size, size=batch_size)
        return decompress_planes(self.planes[ix]), self.pol_idx[ix], self.pol_p[ix].astype(np.float32), self.wdl[ix]

    def save(self, path: Path):
        n = self.size
        np.savez(path, planes=self.planes[:n], pol_idx=self.pol_idx[:n], pol_p=self.pol_p[:n],
                 wdl=self.wdl[:n], pos=self.pos, total_added=self.total_added)

    def load(self, path: Path):
        d = np.load(path)
        n = min(len(d["wdl"]), self.capacity)
        self.planes[:n], self.pol_idx[:n] = d["planes"][:n], d["pol_idx"][:n]
        self.pol_p[:n], self.wdl[:n] = d["pol_p"][:n], d["wdl"][:n]
        self.size = n
        self.pos = int(d["pos"]) % self.capacity
        self.total_added = int(d["total_added"])
