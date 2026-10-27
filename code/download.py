"""Download Stockfish and the Lichess Elite database.

Usage:
    python download.py stockfish
    python download.py elite
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import Config


def _fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def download_stockfish(cfg: Config):
    target = cfg.paths.stockfish_path
    if target.exists():
        print(f"Stockfish already at {target}")
        return
    zpath = cfg.paths.engines_dir / "stockfish.zip"
    _fetch(cfg.download.stockfish_url, zpath)
    with zipfile.ZipFile(zpath) as z:
        exe = next(n for n in z.namelist() if n.endswith(".exe"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(exe) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zpath.unlink()
    print(f"Stockfish installed at {target}")


def download_elite(cfg: Config):
    out = cfg.paths.warmstart_pgn_dir
    out.mkdir(parents=True, exist_ok=True)
    for month in cfg.download.elite_months:
        pgn = out / f"lichess_elite_{month}.pgn"
        if pgn.exists():
            continue
        zpath = out / f"{month}.zip"
        _fetch(cfg.download.elite_url_template.format(month=month), zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        zpath.unlink()
    print(f"PGN files in {out}")


if __name__ == "__main__":
    cfg = Config()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("stockfish", "all"):
        download_stockfish(cfg)
    if what in ("elite", "all"):
        download_elite(cfg)
