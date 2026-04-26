import pygame
import sys
from typing import Optional

from core.board import Board, Player, Piece, PieceType, Move, create_initial_board
from core.move_validator import MoveValidator
from game.game_manager import GameManager, GameState
from api.deepseek_client import DeepSeekClient
from config.settings import get_deepseek_config


WINDOW_WIDTH = 720
WINDOW_HEIGHT = 820
BOARD_MARGIN = 40
CELL_SIZE = 70

COLOR_BG = (245, 222, 179)
COLOR_BOARD = (222, 184, 135)
COLOR_LINE = (139, 69, 19)
COLOR_RED = (200, 0, 0)
COLOR_BLACK = (20, 20, 20)
COLOR_SELECTED = (0, 200, 0)
COLOR_VALID_MOVE = (0, 150, 255)
COLOR_LAST_MOVE = (255, 165, 0)


class PygameUI:
    def __init__(self, game_manager: GameManager):
        pygame.init()
        pygame.display.set_caption("中国象棋 - DeepSeek AI")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.game = game_manager
        self.clock = pygame.time.Clock()

        self.selected_piece: Optional[tuple] = None
        self.valid_moves: list = []
        self.last_move: Optional[Move] = None

        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 72)
        self.small_font = pygame.font.Font(None, 24)

        self._chinese_font = self._get_chinese_font()

        self.ai_thinking = False
        self.ai_message = ""

    def _get_chinese_font(self, size=72):
        font_names = ["SimHei", "Microsoft YaHei", "KaiTi", "FangSong", "Arial Unicode MS"]
        for font_name in font_names:
            try:
                return pygame.font.SysFont(font_name, size, False)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    def _get_chinese_font_small(self, size=30):
        return self._get_chinese_font(size)

    def board_to_screen(self, row, col):
        x = BOARD_MARGIN + col * CELL_SIZE
        y = BOARD_MARGIN + row * CELL_SIZE
        return x, y

    def screen_to_board(self, x, y):
        col = int((x - BOARD_MARGIN + CELL_SIZE / 2) / CELL_SIZE)
        row = int((y - BOARD_MARGIN + CELL_SIZE / 2) / CELL_SIZE)
        if 0 <= row < 10 and 0 <= col < 9:
            return row, col
        return None, None

    def draw_board(self):
        self.screen.fill(COLOR_BG)

        for i in range(9):
            start_x = BOARD_MARGIN + i * CELL_SIZE
            end_x = start_x
            start_y = BOARD_MARGIN
            end_y = BOARD_MARGIN + 9 * CELL_SIZE
            if i == 0 or i == 8:
                pygame.draw.line(self.screen, COLOR_LINE, (start_x, start_y), (end_x, end_y), 2)
            else:
                pygame.draw.line(self.screen, COLOR_LINE, (start_x, start_y), (end_x, end_y), 1)

        for i in range(10):
            start_x = BOARD_MARGIN
            end_x = BOARD_MARGIN + 8 * CELL_SIZE
            start_y = BOARD_MARGIN + i * CELL_SIZE
            end_y = start_y
            pygame.draw.line(self.screen, COLOR_LINE, (start_x, start_y), (end_x, end_y), 1 if i != 0 and i != 9 else 2)

        palace_points = [
            (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 2 * CELL_SIZE),
            (BOARD_MARGIN + 7 * CELL_SIZE, BOARD_MARGIN + 2 * CELL_SIZE),
            (BOARD_MARGIN + 7 * CELL_SIZE, BOARD_MARGIN),
            (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN),
        ]
        for i in range(3):
            pygame.draw.line(self.screen, COLOR_LINE,
                           (palace_points[i * 2][0], palace_points[i * 2][1]),
                           (palace_points[i * 2 + 1][0], palace_points[i * 2 + 1][1]), 1)

        for i in range(3):
            pygame.draw.line(self.screen, COLOR_LINE,
                           (BOARD_MARGIN + (8 - i * 2) * CELL_SIZE, BOARD_MARGIN + 7 * CELL_SIZE),
                           (BOARD_MARGIN + (8 - (i * 2 + 1)) * CELL_SIZE, BOARD_MARGIN + 9 * CELL_SIZE), 1)

        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN + 4 * CELL_SIZE),
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 4 * CELL_SIZE), 1)
        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN + 5 * CELL_SIZE),
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 5 * CELL_SIZE), 1)

        font = self._get_chinese_font_small(30)
        chinese_nums = "九八七六五四三二一"
        for i in range(9):
            text = font.render(chinese_nums[i], True, COLOR_LINE)
            self.screen.blit(text, (BOARD_MARGIN + i * CELL_SIZE - 8, BOARD_MARGIN - 30))

        for i in range(10):
            text = font.render(str(i), True, COLOR_LINE)
            self.screen.blit(text, (BOARD_MARGIN - 25, BOARD_MARGIN + i * CELL_SIZE - 10))

        text = self._get_chinese_font_small(28).render("楚河", True, COLOR_LINE)
        self.screen.blit(text, (BOARD_MARGIN + 2 * CELL_SIZE, BOARD_MARGIN + 4 * CELL_SIZE + 15))
        text = self._get_chinese_font_small(28).render("汉界", True, COLOR_LINE)
        self.screen.blit(text, (BOARD_MARGIN + 6 * CELL_SIZE, BOARD_MARGIN + 4 * CELL_SIZE + 15))

    def draw_pieces(self):
        for row in range(10):
            for col in range(9):
                piece = self.game.board.get_piece(row, col)
                if piece:
                    x, y = self.board_to_screen(row, col)
                    self._draw_piece(x, y, piece)

    def _draw_piece(self, x, y, piece: Piece):
        radius = CELL_SIZE // 2 - 5
        color = COLOR_RED if piece.player == Player.RED else COLOR_BLACK

        if self.selected_piece and self.selected_piece == (self._get_piece_row_col(piece)):
            pygame.draw.circle(self.screen, COLOR_SELECTED, (x, y), radius + 4, 3)

        pygame.draw.circle(self.screen, COLOR_BOARD, (x, y), radius)
        pygame.draw.circle(self.screen, color, (x, y), radius, 2)

        text = self._chinese_font.render(piece.name, True, color)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)

    def _get_piece_row_col(self, piece: Piece) -> Optional[tuple]:
        for row in range(10):
            for col in range(9):
                if self.game.board.get_piece(row, col) == piece:
                    return (row, col)
        return None

    def draw_valid_moves(self):
        for move in self.valid_moves:
            _, (to_row, to_col) = move
            x, y = self.board_to_screen(to_row, to_col)
            pygame.draw.circle(self.screen, COLOR_VALID_MOVE, (x, y), 8, 2)

    def draw_last_move(self):
        if self.last_move:
            for pos in self.last_move:
                row, col = pos
                x, y = self.board_to_screen(row, col)
                pygame.draw.circle(self.screen, COLOR_LAST_MOVE, (x, y), CELL_SIZE // 2 - 10, 3)

    def draw_ui(self):
        player_text = "红方回合" if self.game.current_player == Player.RED else "黑方回合"
        color = COLOR_RED if self.game.current_player == Player.RED else COLOR_BLACK
        text = self._get_chinese_font_small(28).render(player_text, True, color)
        self.screen.blit(text, (WINDOW_WIDTH // 2 - 40, 15))

        if self.ai_thinking:
            text = self._get_chinese_font_small(28).render("AI 思考中...", True, (100, 100, 100))
            self.screen.blit(text, (WINDOW_WIDTH // 2 - 50, 45))

        if self.ai_message:
            text = self._get_chinese_font_small(24).render(self.ai_message[:50], True, (50, 50, 50))
            self.screen.blit(text, (WINDOW_WIDTH // 2 - 150, 75))

        btn_restart = pygame.Rect(WINDOW_WIDTH - 100, 15, 80, 35)
        pygame.draw.rect(self.screen, (200, 200, 200), btn_restart)
        text = self._get_chinese_font_small(24).render("重新开始", True, (0, 0, 0))
        self.screen.blit(text, (WINDOW_WIDTH - 90, 22))
        self.btn_restart = btn_restart

    def draw_game_over(self):
        if self.game.state == GameState.GAME_OVER:
            winner = "红方" if self.game.winner == Player.RED else "黑方"
            text = self._get_chinese_font(72).render(f"{winner}获胜!", True, COLOR_RED)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            pygame.draw.rect(self.screen, COLOR_BG, text_rect.inflate(40, 40))
            self.screen.blit(text, text_rect)

    def handle_click(self, pos):
        if self.btn_restart.collidepoint(pos):
            self.game.start()
            self.selected_piece = None
            self.valid_moves = []
            self.last_move = None
            return

        if self.game.state != GameState.PLAYING:
            return

        if self.game.current_player != Player.RED:
            return

        row, col = self.screen_to_board(*pos)
        if row is None:
            return

        for move in self.valid_moves:
            if move[1] == (row, col):
                self.game.make_human_move(move)
                self.last_move = move
                self.selected_piece = None
                self.valid_moves = []
                return

        piece = self.game.board.get_piece(row, col)
        if piece and piece.player == Player.RED:
            self.selected_piece = (row, col)
            self.valid_moves = [
                m for m in self.game.get_valid_moves()
                if m[0] == (row, col) and self.game.validator.validate_move(m, Player.RED)
            ]
        else:
            self.selected_piece = None
            self.valid_moves = []

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.draw_board()
            self.draw_last_move()
            self.draw_valid_moves()
            self.draw_pieces()
            self.draw_ui()
            self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(30)

            if self.game.state == GameState.PLAYING and self.game.current_player == Player.BLACK and not self.ai_thinking:
                self.ai_thinking = True
                response = self.game.make_ai_move()
                if response:
                    self.last_move = response.move
                    if response.reasoning:
                        self.ai_message = response.reasoning[:80]
                self.ai_thinking = False

        pygame.quit()
        sys.exit()


def main():
    config = get_deepseek_config()

    if not config.api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量")
        print("在项目根目录创建 .env 文件，内容: DEEPSEEK_API_KEY=your_api_key")
        sys.exit(1)

    deepseek_client = DeepSeekClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model
    )

    game_manager = GameManager(
        deepseek_client=deepseek_client,
        first_player=Player.RED
    )
    game_manager.start()

    ui = PygameUI(game_manager)
    ui.run()


if __name__ == "__main__":
    main()