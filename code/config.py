"""Central configuration. All parameters and hyperparameters live here.

Edit the defaults below, or build a modified copy in a script, e.g.
    cfg = Config()
    cfg.mcts.n_simulations = 64
"""
import platform
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (parent of code/)
IS_WINDOWS = platform.system() == "Windows"


def _stockfish_asset() -> str:
    """Release asset for this OS / CPU (https://github.com/official-stockfish/Stockfish/releases)."""
    system = platform.system()
    arm = platform.machine().lower() in ("arm64", "aarch64")
    if system == "Windows":
        return f"stockfish-windows-{'arm64' if arm else 'x86-64'}-universal.zip"
    if system == "Darwin":
        return "stockfish-macos-universal.tar.gz"  # one binary for Apple Silicon and Intel
    return f"stockfish-linux-{'arm64' if arm else 'x86-64'}-universal.tar.gz"


@dataclass
class PathConfig:
    data_dir: Path = ROOT / "data"
    engines_dir: Path = ROOT / "engines"
    runs_dir: Path = ROOT / "runs"
    stockfish_path: Path = ROOT / "engines" / "stockfish" / ("stockfish.exe" if IS_WINDOWS else "stockfish")
    warmstart_pgn_dir: Path = ROOT / "data" / "lichess_elite"
    warmstart_dataset: Path = ROOT / "data" / "warmstart.npz"


@dataclass
class DownloadConfig:
    stockfish_url: str = (
        "https://github.com/official-stockfish/Stockfish/releases/latest/download/" + _stockfish_asset()
    )
    # Lichess Elite Database (https://database.nikonoel.fr), one zip per month.
    elite_url_template: str = "https://database.nikonoel.fr/lichess_elite_{month}.zip"
    elite_months: tuple = ("2023-01", "2023-02", "2023-03")


@dataclass
class ModelConfig:
    in_planes: int = 20          # see chess_env/encoding.py
    n_blocks: int = 8            # residual blocks
    channels: int = 96           # filters per conv layer
    policy_channels: int = 32
    value_hidden: int = 128
    n_moves: int = 4168          # 64*64 from-to + 72 underpromotions


@dataclass
class DeviceConfig:
    device: str = "auto"         # "auto" (xpu -> cuda -> mps -> cpu), "xpu", "cuda", "mps", "cpu"
    torch_threads: int = 4       # intra-op threads on CPU


@dataclass
class MCTSConfig:
    n_simulations: int = 32      # simulations per move (full search)
    max_considered_actions: int = 16   # m in Gumbel sequential halving
    c_visit: float = 50.0        # sigma transform constants (Danihelka et al. 2022)
    c_scale: float = 1.0
    gumbel_scale: float = 1.0    # 0 disables Gumbel noise (deterministic play)


@dataclass
class SelfPlayConfig:
    n_parallel_games: int = 64   # games stepped together for batched inference
    games_per_iteration: int = 128
    # Playout cap randomisation (KataGo): only full searches become policy targets
    full_search_prob: float = 0.25
    fast_simulations: int = 8
    gumbel_plies: int = 30       # plies with Gumbel noise at root (exploration)
    max_plies: int = 300         # adjudicate draw after this many plies
    resign_threshold: float = -0.90   # value (W-L) below which a side may resign
    resign_consecutive: int = 4
    no_resign_fraction: float = 0.1   # games where resignation is disabled


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    buffer_size: int = 300_000   # positions in replay buffer
    min_buffer_size: int = 10_000
    train_steps_per_iteration: int = 200
    checkpoint_every: int = 1    # iterations between overwriting latest.pt
    snapshot_every: int = 10     # iterations between kept checkpoints (for Elo progress)
    buffer_save_every: int = 10  # iterations between saving the replay buffer (~400MB)
    n_iterations: int = 100_000  # effectively "run until stopped"
    seed: int = 0


@dataclass
class WarmStartConfig:
    min_elo: int = 2000          # both players at least this rating
    max_games: int = 50_000      # ~80 positions/game, ~1.3KB each -> ~5GB in RAM
    skip_first_plies: int = 6    # opening positions are low-information
    epochs: int = 3
    batch_size: int = 1024
    lr: float = 2e-3
    val_fraction: float = 0.02


@dataclass
class EvalConfig:
    n_games: int = 100
    sims: int = 64               # agent simulations per move in evaluation
    opponent_elo: int = 1400     # Stockfish UCI_Elo (min ~1320)
    opponent_skill_level: int = -1   # >=0 uses "Skill Level" instead of UCI_Elo
    opponent_time: float = 0.05  # seconds per move for Stockfish
    max_plies: int = 300
    random_opening_plies: int = 4    # randomised openings to avoid repeated games


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    warmstart: WarmStartConfig = field(default_factory=WarmStartConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "main"

    @property
    def run_dir(self) -> Path:
        return self.paths.runs_dir / self.run_name
