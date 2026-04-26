from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


class PieceType(Enum):
    KING = 'k'
    ADVISOR = 'a'
    ELEPHANT = 'e'
    HORSE = 'h'
    ROOK = 'r'
    CANNON = 'c'
    PAWN = 'p'


class Player(Enum):
    RED = 1
    BLACK = -1


@dataclass
class Piece:
    piece_type: PieceType
    player: Player

    @property
    def name(self) -> str:
        names = {
            (PieceType.KING, Player.RED): '帅',
            (PieceType.KING, Player.BLACK): '将',
            (PieceType.ADVISOR, Player.RED): '仕',
            (PieceType.ADVISOR, Player.BLACK): '士',
            (PieceType.ELEPHANT, Player.RED): '相',
            (PieceType.ELEPHANT, Player.BLACK): '象',
            (PieceType.HORSE, Player.RED): '马',
            (PieceType.HORSE, Player.BLACK): '马',
            (PieceType.ROOK, Player.RED): '车',
            (PieceType.ROOK, Player.BLACK): '车',
            (PieceType.CANNON, Player.RED): '炮',
            (PieceType.CANNON, Player.BLACK): '炮',
            (PieceType.PAWN, Player.RED): '兵',
            (PieceType.PAWN, Player.BLACK): '卒',
        }
        return names.get((self.piece_type, self.player), '?')


Position = Tuple[int, int]
Move = Tuple[Position, Position]


@dataclass
class Board:
    squares: List[List[Optional[Piece]]] = field(
        default_factory=lambda: [[None] * 9 for _ in range(10)]
    )

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < 10 and 0 <= col < 9:
            return self.squares[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]):
        self.squares[row][col] = piece

    def to_text_representation(self, perspective: Player = Player.RED) -> str:
        lines = []
        for row in range(10):
            row_str = ""
            for col in range(9):
                piece = self.squares[row][col]
                if piece is None:
                    row_str += "　"
                else:
                    sign = "+" if piece.player == Player.RED else "-"
                    row_str += f"{sign}{piece.piece_type.value}"
                row_str += " "
            lines.append(row_str)
        return "\n".join(lines)

    def copy(self) -> 'Board':
        new_board = Board()
        for r in range(10):
            for c in range(9):
                new_board.squares[r][c] = self.squares[r][c]
        return new_board


def create_initial_board() -> Board:
    board = Board()

    # Black pieces (top, rows 0-2)
    board.set_piece(0, 0, Piece(PieceType.ROOK, Player.BLACK))
    board.set_piece(0, 1, Piece(PieceType.HORSE, Player.BLACK))
    board.set_piece(0, 2, Piece(PieceType.ELEPHANT, Player.BLACK))
    board.set_piece(0, 3, Piece(PieceType.ADVISOR, Player.BLACK))
    board.set_piece(0, 4, Piece(PieceType.KING, Player.BLACK))
    board.set_piece(0, 5, Piece(PieceType.ADVISOR, Player.BLACK))
    board.set_piece(0, 6, Piece(PieceType.ELEPHANT, Player.BLACK))
    board.set_piece(0, 7, Piece(PieceType.HORSE, Player.BLACK))
    board.set_piece(0, 8, Piece(PieceType.ROOK, Player.BLACK))
    board.set_piece(2, 1, Piece(PieceType.CANNON, Player.BLACK))
    board.set_piece(2, 7, Piece(PieceType.CANNON, Player.BLACK))
    board.set_piece(3, 0, Piece(PieceType.PAWN, Player.BLACK))
    board.set_piece(3, 2, Piece(PieceType.PAWN, Player.BLACK))
    board.set_piece(3, 4, Piece(PieceType.PAWN, Player.BLACK))
    board.set_piece(3, 6, Piece(PieceType.PAWN, Player.BLACK))
    board.set_piece(3, 8, Piece(PieceType.PAWN, Player.BLACK))

    # Red pieces (bottom, rows 7-9)
    board.set_piece(9, 0, Piece(PieceType.ROOK, Player.RED))
    board.set_piece(9, 1, Piece(PieceType.HORSE, Player.RED))
    board.set_piece(9, 2, Piece(PieceType.ELEPHANT, Player.RED))
    board.set_piece(9, 3, Piece(PieceType.ADVISOR, Player.RED))
    board.set_piece(9, 4, Piece(PieceType.KING, Player.RED))
    board.set_piece(9, 5, Piece(PieceType.ADVISOR, Player.RED))
    board.set_piece(9, 6, Piece(PieceType.ELEPHANT, Player.RED))
    board.set_piece(9, 7, Piece(PieceType.HORSE, Player.RED))
    board.set_piece(9, 8, Piece(PieceType.ROOK, Player.RED))
    board.set_piece(7, 1, Piece(PieceType.CANNON, Player.RED))
    board.set_piece(7, 7, Piece(PieceType.CANNON, Player.RED))
    board.set_piece(6, 0, Piece(PieceType.PAWN, Player.RED))
    board.set_piece(6, 2, Piece(PieceType.PAWN, Player.RED))
    board.set_piece(6, 4, Piece(PieceType.PAWN, Player.RED))
    board.set_piece(6, 6, Piece(PieceType.PAWN, Player.RED))
    board.set_piece(6, 8, Piece(PieceType.PAWN, Player.RED))

    return board