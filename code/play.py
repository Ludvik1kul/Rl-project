"""Play chess against an agent checkpoint, its raw policy, or Stockfish in the browser.

Usage:
    python play.py            # then open http://127.0.0.1:5000

Defaults (port, device, simulations, time control, Stockfish Elo) are in PlayConfig.
The board (chessground) and rules/clock logic (chess.js) run in the page; the server
only replays the move list and returns the opponent's move.
"""
import time
from dataclasses import replace
from pathlib import Path

import chess
import chess.engine
import numpy as np
from flask import Flask, jsonify, request, send_file

from chess_env.encoding import encode_board, legal_move_indices
from config import Config
from mcts.gumbel import gumbel_search, run_batched
from model import ChessNet, Evaluator, get_device, load_checkpoint

UI_FILE = Path(__file__).resolve().parent / "play_ui.html"


class Opponents:
    """Lazily loaded checkpoints (cached by path) and one shared Stockfish process."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(replace(cfg.device, device=cfg.play.device))
        self.evaluators = {}
        self.engine = None
        self.rng = np.random.default_rng(0)

    def checkpoints(self):
        root = self.cfg.paths.runs_dir
        return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.pt"))

    def evaluator(self, ckpt: str) -> Evaluator:
        if ckpt not in self.checkpoints():
            raise ValueError(f"unknown checkpoint {ckpt}")
        if ckpt not in self.evaluators:
            net = ChessNet(self.cfg.model)
            load_checkpoint(self.cfg.paths.runs_dir / ckpt, net, map_location=self.device)
            self.evaluators[ckpt] = Evaluator(net, self.device)
        return self.evaluators[ckpt]

    def agent_move(self, board, ckpt, sims):
        ev = self.evaluator(ckpt)
        if sims <= 0:  # raw policy: network's most likely legal move, no search
            moves, idx = legal_move_indices(board)
            logits, wdl = ev(encode_board(board)[None])
            return moves[int(np.argmax(logits[0][idx]))], wdl[0]
        gen = gumbel_search(board, sims, self.cfg.mcts, self.cfg.play.gumbel_scale, self.rng)
        res = run_batched(ev, [gen])[0]
        return res.move, res.root.wdl

    def stockfish_move(self, board, elo):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.cfg.paths.stockfish_path))
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        res = self.engine.play(board, chess.engine.Limit(time=self.cfg.play.stockfish_time))
        return res.move, None

    def close(self):
        if self.engine is not None:
            self.engine.quit()


def create_app(cfg: Config) -> Flask:
    app = Flask(__name__)
    opp = Opponents(cfg)
    app.config["opponents"] = opp

    @app.get("/")
    def index():
        return send_file(UI_FILE)

    @app.get("/api/options")
    def options():
        pc = cfg.play
        return jsonify(checkpoints=opp.checkpoints(), sims=pc.sims, minutes=pc.minutes,
                       increment=pc.increment, stockfish_elo=pc.stockfish_elo)

    @app.post("/api/move")
    def move():
        req = request.get_json()
        board = chess.Board()
        for uci in req["moves"]:
            board.push_uci(uci)
        if board.is_game_over(claim_draw=True):
            return jsonify(error="game is over"), 400
        t0 = time.perf_counter()
        if req["opponent"] == "stockfish":
            mv, wdl = opp.stockfish_move(board, int(req["elo"]))
        else:
            sims = 0 if req["opponent"] == "policy" else int(req["sims"])
            mv, wdl = opp.agent_move(board, req["ckpt"], sims)
        out = {"move": mv.uci(), "seconds": time.perf_counter() - t0}
        if wdl is not None:  # from the mover's perspective
            out["wdl"] = [float(x) for x in wdl]
        return jsonify(out)

    return app


if __name__ == "__main__":
    cfg = Config()
    app = create_app(cfg)
    print(f"Open http://{cfg.play.host}:{cfg.play.port}")
    try:
        app.run(host=cfg.play.host, port=cfg.play.port, threaded=False)
    finally:
        app.config["opponents"].close()
