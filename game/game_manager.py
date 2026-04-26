from typing import Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from core.board import Board, Player, Piece, PieceType, Move, create_initial_board
from core.move_validator import MoveValidator
from api.deepseek_client import DeepSeekClient, AIAgentResponse


class GameState(Enum):
    INIT = "init"
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class GameManager:
    board: Board
    current_player: Player = Player.RED
    state: GameState = GameState.INIT
    winner: Optional[Player] = None
    move_history: List[str] = field(default_factory=list)

    def __init__(
        self,
        deepseek_client: Optional[DeepSeekClient] = None,
        first_player: Player = Player.RED
    ):
        self.board = create_initial_board()
        self.current_player = first_player
        self.state = GameState.INIT
        self.winner = None
        self.move_history = []
        self.api_client = deepseek_client
        self.validator = MoveValidator(self.board)

        self.on_game_over: Optional[Callable[[Player], None]] = None
        self.on_move_made: Optional[Callable[[Move], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def start(self):
        self.board = create_initial_board()
        self.current_player = Player.RED
        self.state = GameState.PLAYING
        self.winner = None
        self.move_history = []

    def get_valid_moves(self) -> List[Move]:
        moves = []
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.player == self.current_player:
                    piece_moves = self._get_piece_moves(row, col, piece)
                    moves.extend(piece_moves)
        return moves

    def _get_piece_moves(self, row: int, col: int, piece: Piece) -> List[Move]:
        moves = []
        pt = piece.piece_type

        if pt == PieceType.KING:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if self.validator.is_valid_position(nr, nc):
                    moves.append(((row, col), (nr, nc)))

        elif pt == PieceType.ADVISOR:
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = row + dr, col + dc
                if self.validator.is_valid_position(nr, nc):
                    moves.append(((row, col), (nr, nc)))

        elif pt == PieceType.ELEPHANT:
            for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
                nr, nc = row + dr, col + dc
                if self.validator.is_valid_position(nr, nc):
                    moves.append(((row, col), (nr, nc)))

        elif pt == PieceType.HORSE:
            for dr, dc, br, bc in [
                (-2, -1, -1, 0), (-2, 1, -1, 0),
                (2, -1, 1, 0), (2, 1, 1, 0),
                (-1, -2, 0, -1), (1, -2, 0, -1),
                (-1, 2, 0, 1), (1, 2, 0, 1)
            ]:
                nr, nc = row + dr, col + dc
                br_row, br_col = row + br, col + bc
                if self.validator.is_valid_position(nr, nc):
                    moves.append(((row, col), (nr, nc)))

        elif pt == PieceType.ROOK:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for i in range(1, 10):
                    nr, nc = row + dr * i, col + dc * i
                    if not self.validator.is_valid_position(nr, nc):
                        break
                    moves.append(((row, col), (nr, nc)))
                    if self.board.get_piece(nr, nc) is not None:
                        break

        elif pt == PieceType.CANNON:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                jumped = False
                for i in range(1, 10):
                    nr, nc = row + dr * i, col + dc * i
                    if not self.validator.is_valid_position(nr, nc):
                        break
                    if not jumped:
                        moves.append(((row, col), (nr, nc)))
                        if self.board.get_piece(nr, nc) is not None:
                            jumped = True
                    else:
                        if self.board.get_piece(nr, nc) is not None:
                            moves.append(((row, col), (nr, nc)))
                            break

        elif pt == PieceType.PAWN:
            direction = -1 if piece.player == Player.RED else 1
            nr, nc = row + direction, col
            if self.validator.is_valid_position(nr, nc):
                moves.append(((row, col), (nr, nc)))
            if (piece.player == Player.RED and row <= 4) or \
               (piece.player == Player.BLACK and row >= 5):
                for dc in [-1, 1]:
                    nc = col + dc
                    if self.validator.is_valid_position(row, nc):
                        moves.append(((row, col), (row, nc)))

        return moves

    def make_human_move(self, move: Move) -> bool:
        if self.state != GameState.PLAYING:
            return False

        player_name = "红方" if self.current_player == Player.RED else "黑方"

        valid_moves = self.get_valid_moves()
        if not self.validator.validate_move(move, self.current_player):
            if self.on_error:
                self.on_error(f"{player_name}走棋不合法")
            return False

        self._execute_move(move)
        return True

    def make_ai_move(self) -> Optional[AIAgentResponse]:
        if self.state != GameState.PLAYING:
            return None

        if self.api_client is None:
            return None

        valid_moves = self.get_valid_moves()

        response = self.api_client.get_ai_move(
            self.board,
            self.current_player,
            valid_moves,
            self.move_history
        )

        if response.move and response.is_valid:
            self._execute_move(response.move)

        return response

    def _execute_move(self, move: Move):
        (from_row, from_col), (to_row, to_col) = move
        piece = self.board.get_piece(from_row, from_col)
        captured = self.board.get_piece(to_row, to_col)

        self.board.set_piece(to_row, to_col, piece)
        self.board.set_piece(from_row, from_col, None)

        move_text = self._format_move_text(move, piece, captured)
        self.move_history.append(move_text)

        if captured and captured.piece_type == PieceType.KING:
            self.state = GameState.GAME_OVER
            self.winner = self.current_player
            if self.on_game_over:
                self.on_game_over(self.winner)
        else:
            self.current_player = Player.BLACK if self.current_player == Player.RED else Player.RED

            if self.validator.is_in_check(self.current_player):
                if self.on_error:
                    player_name = "红方" if self.current_player == Player.RED else "黑方"
                    if self._has_valid_moves():
                        self.on_error(f"{player_name}被将军！")
                    else:
                        self.state = GameState.GAME_OVER
                        self.winner = Player.BLACK if self.current_player == Player.RED else Player.RED
                        if self.on_game_over:
                            self.on_game_over(self.winner)

        if self.on_move_made:
            self.on_move_made(move)

    def _has_valid_moves(self) -> bool:
        valid_moves = self.get_valid_moves()
        for move in valid_moves:
            if self.validator.validate_move(move, self.current_player):
                return True
        return False

    def _format_move_text(self, move: Move, piece: Piece, captured: Optional[Piece]) -> str:
        (from_row, from_col), (to_row, to_col) = move

        col_names = "九八七六五四三二一"
        row_names = "零一二三四五六七八九"
        piece_name = piece.name
        if piece.player == Player.RED:
            from_pos = f"{col_names[from_col]}路{row_names[from_row]}"
            to_pos = f"{col_names[to_col]}路{row_names[to_row]}"
        else:
            from_pos = f"{col_names[from_col]}路{row_names[9 - from_row]}"
            to_pos = f"{col_names[to_col]}路{row_names[9 - to_row]}"

        action = "吃" if captured else "到"

        return f"{piece_name}{from_pos}{action}{to_pos}"