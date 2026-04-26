from core.board import Board, Piece, PieceType, Player, Position, Move


class MoveValidator:
    def __init__(self, board: Board):
        self.board = board

    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 10 and 0 <= col < 9

    def in_palace(self, row: int, col: int, player: Player) -> bool:
        if player == Player.RED:
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5

    def is_friendly(self, row: int, col: int, player: Player) -> bool:
        piece = self.board.get_piece(row, col)
        return piece is not None and piece.player == player

    def is_enemy(self, row: int, col: int, player: Player) -> bool:
        piece = self.board.get_piece(row, col)
        return piece is not None and piece.player != player

    def validate_move(self, move: Move, player: Player) -> bool:
        (from_row, from_col), (to_row, to_col) = move

        piece = self.board.get_piece(from_row, from_col)
        if piece is None or piece.player != player:
            return False

        if self.is_friendly(to_row, to_col, player):
            return False

        if not self._check_piece_move_rules(piece, from_row, from_col, to_row, to_col):
            return False

        if self._would_cause_suicide(move, player):
            return False

        return True

    def _check_piece_move_rules(self, piece: Piece, fr: int, fc: int, tr: int, tc: int) -> bool:
        dr, dc = tr - fr, tc - fc

        if piece.piece_type == PieceType.KING:
            return self._check_king_move(dr, dc, piece.player)
        elif piece.piece_type == PieceType.ADVISOR:
            return self._check_advisor_move(dr, dc, piece.player)
        elif piece.piece_type == PieceType.ELEPHANT:
            return self._check_elephant_move(fr, fc, dr, dc, piece.player)
        elif piece.piece_type == PieceType.HORSE:
            return self._check_horse_move(fr, fc, dr, dc)
        elif piece.piece_type == PieceType.ROOK:
            return self._check_rook_move(fr, fc, tr, tc)
        elif piece.piece_type == PieceType.CANNON:
            return self._check_cannon_move(fr, fc, tr, tc)
        elif piece.piece_type == PieceType.PAWN:
            return self._check_pawn_move(fr, fc, dr, dc, piece.player)

        return False

    def _check_king_move(self, dr: int, dc: int, player: Player) -> bool:
        if abs(dr) + abs(dc) == 1:
            return True
        return False

    def _check_advisor_move(self, dr: int, dc: int, player: Player) -> bool:
        if abs(dr) == 1 and abs(dc) == 1:
            return True
        return False

    def _check_elephant_move(self, fr: int, fc: int, dr: int, dc: int, player: Player) -> bool:
        if abs(dr) == 2 and abs(dc) == 2:
            block_row, block_col = fr + dr // 2, fc + dc // 2
            if self.board.get_piece(block_row, block_col) is None:
                if player == Player.RED and fr > 4:
                    return True
                elif player == Player.BLACK and fr < 5:
                    return True
        return False

    def _check_horse_move(self, fr: int, fc: int, dr: int, dc: int) -> bool:
        if (abs(dr), abs(dc)) == (2, 1) or (abs(dr), abs(dc)) == (1, 2):
            block_r, block_c = None, None
            if dr == 2 or dr == -2:
                block_r, block_c = fr + dr // 2, fc
            else:
                block_r, block_c = fr, fc + dc // 2
            if block_r is not None and self.board.get_piece(block_r, block_c) is None:
                return True
        return False

    def _check_rook_move(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        if fr == tr:
            step = 1 if tc > fc else -1
            for c in range(fc + step, tc, step):
                if self.board.get_piece(fr, c) is not None:
                    return False
            return True
        elif tc == fc:
            step = 1 if tr > fr else -1
            for r in range(fr + step, tr, step):
                if self.board.get_piece(r, fc) is not None:
                    return False
            return True
        return False

    def _check_cannon_move(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        if fr == tr:
            step = 1 if tc > fc else -1
            count = 0
            for c in range(fc + step, tc, step):
                if self.board.get_piece(fr, c) is not None:
                    count += 1
            target = self.board.get_piece(fr, tc)
            if target is None:
                return count == 0
            else:
                return count == 1
        elif tc == fc:
            step = 1 if tr > fr else -1
            count = 0
            for r in range(fr + step, tr, step):
                if self.board.get_piece(r, fc) is not None:
                    count += 1
            target = self.board.get_piece(tr, tc)
            if target is None:
                return count == 0
            else:
                return count == 1
        return False

    def _check_pawn_move(self, fr: int, fc: int, dr: int, dc: int, player: Player) -> bool:
        direction = -1 if player == Player.RED else 1
        if dr == direction and dc == 0:
            return True
        if (player == Player.RED and fr <= 4) or (player == Player.BLACK and fr >= 5):
            if dr == 0 and abs(dc) == 1:
                return True
        return False

    def _would_cause_suicide(self, move: Move, player: Player) -> bool:
        (from_row, from_col), (to_row, to_col) = move

        from_piece = self.board.get_piece(from_row, from_col)
        to_piece = self.board.get_piece(to_row, to_col)

        self.board.set_piece(to_row, to_col, from_piece)
        self.board.set_piece(from_row, from_col, None)

        in_check = self.is_in_check(player)

        self.board.set_piece(from_row, from_col, from_piece)
        self.board.set_piece(to_row, to_col, to_piece)

        return in_check

    def is_in_check(self, player: Player) -> bool:
        king_pos = None
        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.piece_type == PieceType.KING and piece.player == player:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if not king_pos:
            return True

        opponent = Player.BLACK if player == Player.RED else Player.RED

        for row in range(10):
            for col in range(9):
                piece = self.board.get_piece(row, col)
                if piece and piece.player == opponent:
                    if self._can_attack((row, col), king_pos, piece):
                        return True
        return False

    def _can_attack(self, from_pos: Position, to_pos: Position, piece: Piece) -> bool:
        fr, fc = from_pos
        tr, tc = to_pos
        dr, dc = tr - fr, tc - fc

        if piece.piece_type == PieceType.KING:
            return abs(dr) + abs(dc) == 1 and self.in_palace(tr, tc, piece.player)
        elif piece.piece_type == PieceType.ADVISOR:
            return abs(dr) == 1 and abs(dc) == 1 and self.in_palace(tr, tc, piece.player)
        elif piece.piece_type == PieceType.ELEPHANT:
            return abs(dr) == 2 and abs(dc) == 2 and self._elephant_can_reach(fr, fc, tr, tc, piece.player)
        elif piece.piece_type == PieceType.HORSE:
            return self._check_horse_move(fr, fc, dr, dc)
        elif piece.piece_type == PieceType.ROOK:
            return self._can_rook_reach(fr, fc, tr, tc)
        elif piece.piece_type == PieceType.CANNON:
            return self._can_cannon_reach(fr, fc, tr, tc)
        elif piece.piece_type == PieceType.PAWN:
            direction = -1 if piece.player == Player.RED else 1
            if dr == direction and dc == 0:
                return True
            if (piece.player == Player.RED and fr <= 4) or (piece.player == Player.BLACK and fr >= 5):
                if dr == 0 and abs(dc) == 1:
                    return True
        return False

    def _elephant_can_reach(self, fr: int, fc: int, tr: int, tc: int, player: Player) -> bool:
        if player == Player.RED and tr < 5:
            return False
        if player == Player.BLACK and tr > 4:
            return False
        block_r, block_c = fr + (tr - fr) // 2, fc + (tc - fc) // 2
        return self.board.get_piece(block_r, block_c) is None

    def _can_rook_reach(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        if fr == tr:
            step = 1 if tc > fc else -1
            for c in range(fc + step, tc, step):
                if self.board.get_piece(fr, c) is not None:
                    return False
            return True
        elif tc == fc:
            step = 1 if tr > fr else -1
            for r in range(fr + step, tr, step):
                if self.board.get_piece(r, fc) is not None:
                    return False
            return True
        return False

    def _can_cannon_reach(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        if fr == tr:
            step = 1 if tc > fc else -1
            count = 0
            for c in range(fc + step, tc, step):
                if self.board.get_piece(fr, c) is not None:
                    count += 1
            target = self.board.get_piece(fr, tc)
            return (target is None and count == 0) or (target is not None and count == 1)
        elif tc == fc:
            step = 1 if tr > fr else -1
            count = 0
            for r in range(fr + step, tr, step):
                if self.board.get_piece(r, fc) is not None:
                    count += 1
            target = self.board.get_piece(tr, tc)
            return (target is None and count == 0) or (target is not None and count == 1)
        return False