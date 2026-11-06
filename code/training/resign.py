"""Resignation threshold calibrated on no-resign games (AlphaGo Zero, Silver et al. 2017).

For every (game, side) of a no-resign game we store the lowest threshold that would have made that
side resign: the minimum over all runs of `resign_consecutive` own moves of the largest search value
in the run. A would-be resignation at threshold t is a false positive if the side did not lose.
The threshold is set to the highest value whose false-positive rate stays within the target.
If no threshold within the bounds meets the target, resignation is suspended (threshold RESIGN_OFF)
until one does. With calibration on it also starts suspended: resignation begins only once the
no-resign games have shown a threshold that meets the target.
"""
from collections import deque

import numpy as np

from config import SelfPlayConfig

RESIGN_OFF = -1.0   # search values are >= -1, so `value < RESIGN_OFF` never triggers


def trigger_level(values, k: int):
    """Smallest threshold t such that k consecutive values are all < t (None if fewer than k moves)."""
    if len(values) < k:
        return None
    v = np.asarray(values)
    return float(min(v[i:i + k].max() for i in range(len(v) - k + 1)))


class ResignCalibrator:
    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg
        self.threshold = RESIGN_OFF if cfg.resign_calibrate else cfg.resign_threshold
        self.records = deque(maxlen=cfg.resign_window)   # (trigger level, side lost)

    def add(self, level: float, lost: bool):
        if level is not None:
            self.records.append((level, lost))

    def false_positive_rate(self, threshold: float):
        """(fraction of would-be resignations that did not lose, number of would-be resignations)."""
        hits = [lost for level, lost in self.records if level < threshold]
        return (1.0 - float(np.mean(hits)), len(hits)) if hits else (0.0, 0)

    def update(self):
        if not self.cfg.resign_calibrate:
            return
        lo, hi = self.cfg.resign_bounds
        best = None
        for t in np.linspace(lo, hi, int(round((hi - lo) / 0.01)) + 1):
            fp, n = self.false_positive_rate(t)
            if n >= self.cfg.resign_min_samples and fp <= self.cfg.resign_target_fp:
                best = float(t)
        if best is not None:
            self.threshold = best
        elif self.false_positive_rate(hi)[1] >= self.cfg.resign_min_samples:
            # Enough evidence, yet no threshold down to the lowest bound meets the target: resigning
            # would mislabel games as losses, so stop until the value estimates are reliable again.
            self.threshold = RESIGN_OFF

    def state_dict(self):
        return {"threshold": self.threshold, "records": list(self.records)}

    def load_state_dict(self, state):
        self.threshold = state["threshold"]
        self.records.extend(tuple(r) for r in state["records"])
