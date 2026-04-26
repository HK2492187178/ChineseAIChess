import re
from typing import Optional, List
from core.board import Move


class ResponseParser:
    MOVE_PATTERN = re.compile(r'(\d),(\d)\s*->\s*(\d),(\d)')

    @classmethod
    def parse_move(cls, content: str, valid_moves: List[Move]) -> Optional[Move]:
        move_line_match = re.search(r'走棋[：:]\s*(\d),(\d)\s*->\s*(\d),(\d)', content)
        if move_line_match:
            move = (
                (int(move_line_match.group(1)), int(move_line_match.group(2))),
                (int(move_line_match.group(3)), int(move_line_match.group(4)))
            )
            if cls._is_valid_move(move, valid_moves):
                return move

        for match in cls.MOVE_PATTERN.finditer(content):
            move = (
                (int(match.group(1)), int(match.group(2))),
                (int(match.group(3)), int(match.group(4)))
            )
            if cls._is_valid_move(move, valid_moves):
                return move

        if valid_moves:
            return valid_moves[0]

        return None

    @classmethod
    def _is_valid_move(cls, move: Move, valid_moves: List[Move]) -> bool:
        return move in valid_moves

    @classmethod
    def extract_reasoning(cls, content: str) -> str:
        reasoning_match = re.search(r'思考过程[：:]\s*(.*?)(?=\n走棋|$)', content, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return content[:200].strip()