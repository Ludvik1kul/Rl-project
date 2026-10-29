"""Central configuration. All parameters and hyperparameters live here.

Edit the defaults below, or build a modified copy in a script, e.g.
    cfg = Config()
    cfg.mcts.n_simulations = 64
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root (parent of code/)


@dataclass
class PathConfig:
    data_dir: Path = ROOT / "data"
    engines_dir: Path = ROOT / "engines"
    runs_dir: Path = ROOT / "runs"
    stockfish_path: Path = ROOT / "engines" / "stockfish" / "stockfish.exe"
    warmstart_pgn_dir: Path = ROOT / "data" / "lichess_elite"
    warmstart_dataset: Path = ROOT / "data" / "warmstart.npz"


@dataclass
class DownloadConfig:
    stockfish_url: str = (
        "https://github.com/official-stockfish/Stockfish/releases/latest/download/"
        "stockfish-windows-x86-64-universal.zip"
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
    device: str = "auto"         # "auto" (xpu -> cuda -> cpu), "xpu", "cpu", "cuda"
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
class PlayConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    device: str = "cpu"          # single-game inference is small; keeps the GPU free for training
    sims: int = 64               # default agent simulations per move (0 = raw policy)
    gumbel_scale: float = 0.0    # 0 = deterministic play
    minutes: float = 5.0         # default time control: minutes per side ...
    increment: float = 0.0       # ... plus seconds added after each move
    stockfish_elo: int = 1400    # default Stockfish UCI_Elo (range ~1320-3190)
    stockfish_time: float = 0.1  # seconds per Stockfish move


@dataclass
class ControllerConfig:
    # Actions: 0 play, 1 search deeper, 2 consult Stockfish, 3 consult judge
    base_sims: int = 32
    deeper_multiplier: int = 8   # "search deeper" uses base_sims * this
    expert_elo: int = 2200       # consult-Stockfish strength (raise if agent gets stronger)
    expert_time: float = 0.1
    oracle_depth: int = 16       # privileged training-time evaluator (full strength)
    oracle_threads: int = 2
    mate_score: int = 10_000     # cp value used for forced mates
    # Costs, in the same units as the reward transform (cp by default)
    cost_play: float = 0.0
    cost_deeper: float = 5.0
    cost_expert: float = 20.0
    cost_judge: float = 50.0
    # Reward transform applied to oracle evaluations before taking differences:
    #   "cp": clip(cp, +-cp_clip) | "cp_tanh": cp_scale*tanh(cp/cp_scale)
    #   "expected_score": 1000 * logistic(cp) (Lichess win-prob curve, in milli-points)
    reward_transform: str = "cp"
    cp_clip: float = 1500.0
    cp_scale: float = 600.0
    # Fault injection: the agent's own move (play / deeper) is replaced by a random legal move.
    #   "none" | "uniform": p = fault_rate | "uncertainty": p = min(1, fault_rate*(H_norm/0.5)^power)
    fault_mode: str = "uncertainty"
    fault_rate: float = 0.05
    fault_uncertainty_power: float = 2.0
    # Counterfactual dataset generation
    n_dataset_games: int = 400
    n_parallel_games: int = 16
    sample_prob: float = 0.5     # fraction of positions labelled (others only played through)
    random_opening_plies: int = 6
    max_plies: int = 200
    # Bandit model
    hidden: int = 64
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2    # split by game


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
    play: PlayConfig = field(default_factory=PlayConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    run_name: str = "main"

    @property
    def run_dir(self) -> Path:
        return self.paths.runs_dir / self.run_name
