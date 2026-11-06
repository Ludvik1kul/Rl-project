"""Run logger: JSON lines (one file per stream) plus TensorBoard.

Streams in runs/<run_name>/:
    steps.jsonl   one line per gradient step (losses, accuracies, gradient norm, lr)
    log.jsonl     one line per iteration (self-play, buffer, search and timing statistics)
    eval.jsonl    one line per evaluation (Elo vs Stockfish, match vs previous snapshot, move timing)
    tensorboard/  the same scalars; view with `tensorboard --logdir runs`
"""
import json
import time

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, run_dir):
        self.files = {name: open(run_dir / f"{name}.jsonl", "a") for name in ("steps", "log", "eval")}
        self.tb = SummaryWriter(str(run_dir / "tensorboard"))

    def write(self, stream: str, step: int, entry: dict, tb_prefix: str = None):
        """Appends `entry` to <stream>.jsonl. Numeric fields also go to TensorBoard: keys already
        containing a '/' (e.g. "selfplay/mean_plies") are used as tags, others become <tb_prefix>/<key>."""
        entry = {"step": step, "time": time.time(), **entry}
        self.files[stream].write(json.dumps(entry) + "\n")
        self.files[stream].flush()
        prefix = tb_prefix or stream
        for k, v in entry.items():
            if k not in ("step", "time") and isinstance(v, (int, float)) and not isinstance(v, bool):
                self.tb.add_scalar(k if "/" in k else f"{prefix}/{k}", v, step)
        return entry

    def close(self):
        for f in self.files.values():
            f.close()
        self.tb.close()
