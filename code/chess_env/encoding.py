"""Board and move encodings, always from the side-to-move's perspective.

Board planes (20 x 8 x 8):
    0-5   own pieces  (P, N, B, R, Q, K)
    6-11  opponent pieces
    12    ones (bias plane; the side to move is implicit after flipping)
    13-16 castling rights: own K-side, own Q-side, opp K-side, opp Q-side
    17    en-passant target square
    18    repetition flag (position has occurred before)
    19    halfmove clock / 100

Moves: index = from*64 + to (queen promotions included), plus 72 slots for
underpromotions: 4096 + (from_file*3 + direction)*3 + piece, with
direction in {left, straight, right} and piece in {N, B, R}.
When black is to move, squares are mirrored vertically before encoding.
"""
import chess
import numpy as np

N_PLANES = 20
N_MOVES = 64 * 64 + 8 * 3 * 3
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_UNDERPROMO_INV = {v: k for k, v in _UNDERPROMO.items()}


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    us = board.turn
    flip = us == chess.BLACK
    for sq, piece in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        offset = 0 if piece.color == us else 6
        planes[offset + piece.piece_type - 1, s >> 3, s & 7] = 1.0
    planes[12] = 1.0
    them = not us
    planes[13] = board.has_kingside_castling_rights(us)
    planes[14] = board.has_queenside_castling_rights(us)
    planes[15] = board.has_kingside_castling_rights(them)
    planes[16] = board.has_queenside_castling_rights(them)
    if board.ep_square is not None:
        s = chess.square_mirror(board.ep_square) if flip else board.ep_square
        planes[17, s >> 3, s & 7] = 1.0
    if board.is_repetition(2):
        planes[18] = 1.0
    planes[19] = board.halfmove_clock / 100.0
    return planes


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        direction = (t & 7) - (f & 7) + 1  # 0 left, 1 straight, 2 right
        return 4096 + ((f & 7) * 3 + direction) * 3 + _UNDERPROMO[move.promotion]
    return f * 64 + t


def decode_move(index: int, board: chess.Board) -> chess.Move:
    flip = board.turn == chess.BLACK
    if index >= 4096:
        k = index - 4096
        piece = _UNDERPROMO_INV[k % 3]
        k //= 3
        direction, from_file = k % 3, k // 3
        f = chess.square(from_file, 6)
        t = chess.square(from_file + direction - 1, 7)
        promo = piece
    else:
        f, t = divmod(index, 64)
        promo = None
    if flip:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    if promo is None and board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7):
        promo = chess.QUEEN
    return chess.Move(f, t, promotion=promo)


def legal_move_indices(board: chess.Board):
    """Returns (moves, indices) for all legal moves."""
    moves = list(board.legal_moves)
    turn = board.turn
    return moves, np.fromiter((encode_move(m, turn) for m in moves), dtype=np.int64, count=len(moves))


def outcome_value(result: str, turn: chess.Color) -> float:
    """Game result as a value for the side to move: +1 win, 0 draw, -1 loss."""
    return {0: 1.0, 1: 0.0, 2: -1.0}[outcome_wdl(result, turn)]


def outcome_wdl(result: str, turn: chess.Color) -> int:
    """Game result string -> class index from `turn`'s perspective: 0 win, 1 draw, 2 loss."""
    if result == "1/2-1/2":
        return 1
    white_won = result == "1-0"
    return 0 if white_won == (turn == chess.WHITE) else 2
