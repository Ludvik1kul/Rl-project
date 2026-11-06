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
    warmstart_dataset: Path = ROOT / "data" / "warmstart_sf.npz"   # Stockfish-labelled (warmstart.npz = old)


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
    n_simulations: int = 16      # simulations per move (full search)
    max_considered_actions: int = 16   # m in Gumbel sequential halving (paper: m = min(n, 16))
    c_visit: float = 50.0        # sigma transform constants (Danihelka et al. 2022)
    c_scale: float = 1.0         # paper: 1.0 for Go and chess (0.1 only for Atari)
    # Q-values entering sigma, mapped to [0, 1] (paper App. F):
    #   "fixed"  (q + 1) / 2. Chess values already lie in [-1, 1]; with c_scale = 1 this is the paper's
    #            Go/chess setting. The target moves away from the prior in proportion to the Q gaps.
    #   "minmax" rescales each node's completed Q to [0, 1] (mctx). The paper uses min-max only for Atari,
    #            tree-wide and with c_scale = 0.1. With c_scale = 1 it stretches any gap, even pure noise,
    #            to 51 logits and gives near one-hot targets (test run of Nov 2026: Elo 1409 -> 808).
    q_normalization: str = "fixed"
    gumbel_scale: float = 1.0    # 0 disables Gumbel noise (deterministic play)


@dataclass
class SelfPlayConfig:
    n_parallel_games: int = 64   # games stepped together for batched inference
    games_per_iteration: int = 128
    # Paper: every move is a full search and a training target (full_search_prob = 1).
    # < 1 enables KataGo playout cap randomisation: the other moves use fast_simulations and are not recorded.
    full_search_prob: float = 1.0
    fast_simulations: int = 8
    gumbel_plies: int = 1000     # plies with Gumbel noise at root; >= max_plies = every move (paper)
    # Plies at the start of each game where the move played is sampled in proportion to the root visit
    # counts (MuZero / AlphaGo Zero; paper App. F uses 30, with n = 200). Off by default: with
    # n_simulations = 16 = max_considered_actions every candidate gets one visit, so this would pick
    # uniformly among 16 moves. Gumbel noise already makes each move a sample from the improved policy.
    # Worth trying at 64+ simulations.
    explore_plies: int = 0
    max_plies: int = 300         # adjudicate draw after this many plies
    # Resignation on the root search value (AlphaGo Zero; not used in the Gumbel paper or MuZero).
    # Off by default: in the test run, would-be resignations failed to lose up to 24% of the time even at
    # -0.99, so resigned games became mislabelled losses. Every game is still recorded for calibration,
    # so the false-positive rate stays visible in the logs.
    resign: bool = False
    # When resign is on, the threshold is recalibrated on the no-resign games so that at most
    # resign_target_fp of would-be resignations did not actually lose; if no threshold in resign_bounds
    # meets the target, resignation is suspended until one does.
    resign_threshold: float = -0.90   # initial threshold (search value W-L)
    resign_consecutive: int = 4
    no_resign_fraction: float = 0.1   # with resign on: games where it is disabled (used for calibration)
    resign_calibrate: bool = True
    resign_target_fp: float = 0.05
    resign_min_samples: int = 20      # would-be resignations needed before the threshold moves
    resign_window: int = 2000         # most recent no-resign (game, side) records kept
    resign_bounds: tuple = (-0.99, -0.5)


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    buffer_size: int = 300_000   # positions in replay buffer
    min_buffer_size: int = 10_000
    # Train steps per iteration = target_reuse * new_positions / batch_size, so each position is
    # sampled about target_reuse times while it is in the buffer. 0 -> fixed train_steps_per_iteration.
    target_reuse: float = 6.0
    train_steps_per_iteration: int = 200
    max_train_steps_per_iteration: int = 1000
    # Keep playing batches of games_per_iteration until at least this many new positions (0 = one batch).
    min_positions_per_iteration: int = 0
    checkpoint_every: int = 5    # iterations between saving latest.pt + buffer.npz together (resume point)
    snapshot_every: int = 10     # iterations between kept checkpoints (for Elo progress)
    n_iterations: int = 100_000  # effectively "run until stopped"
    time_limit_hours: float = 0.0    # stop (after a checkpoint) once exceeded; 0 = no limit
    seed: int = 0


@dataclass
class WarmStartConfig:
    min_elo: int = 2000          # both players at least this rating
    max_games: int = 40_000      # games sampled from the PGNs
    positions_per_game: int = 10     # random positions per game (fewer, less correlated positions)
    skip_first_plies: int = 6    # opening positions are low-information
    # Value labels from Stockfish: its WDL estimate (UCI_ShowWDL) as P(win) - P(loss) for the side to move.
    sf_nodes: int = 50_000       # search budget per position
    sf_workers: int = 7          # parallel single-threaded Stockfish processes
    sf_hash_mb: int = 16
    # Value target = sf_value_weight * Stockfish value + (1 - sf_value_weight) * game result (set at training).
    sf_value_weight: float = 1.0
    # After every epoch: a match against Stockfish (0 games = off), logged to runs/<run>/warmstart_logs.
    eval_games: int = 30
    eval_elo: int = 1500         # Stockfish UCI_Elo
    eval_time: float = 0.1       # Stockfish seconds per move
    eval_sims: int = 64          # agent simulations per move
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
    random_opening_plies: int = 4    # each random opening is played twice, colours swapped
    max_parallel_games: int = 50     # games stepped together (agent moves are batched)


@dataclass
class TrainEvalConfig:
    """Periodic evaluation inside train_selfplay.py."""
    every: int = 10              # iterations between evaluations (0 = off)
    at_start: bool = True        # also evaluate the starting weights (iteration 0)
    n_games: int = 40            # vs Stockfish; ~ +-150 Elo 95% CI at 40 games
    sims: int = 64
    opponent_time: float = 0.05
    initial_elo: int = 1600      # first opponent; afterwards the previous estimate is used
    elo_bounds: tuple = (1320, 3190)  # Stockfish UCI_Elo range
    max_elo_step: int = 400      # cap on how far the opponent moves between evaluations
    match_games: int = 20        # vs the snapshot from `every` iterations earlier (0 = off)
    timing_positions: int = 50   # positions used to time a policy-only move


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
    train_eval: TrainEvalConfig = field(default_factory=TrainEvalConfig)
    run_name: str = "v2"         # v2: scalar tanh value head, Stockfish-labelled warm start

    @property
    def run_dir(self) -> Path:
        return self.paths.runs_dir / self.run_name
