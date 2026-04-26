import pygame
import numpy as np
import time
import sys
import copy
import os

# PaddlePaddle是可选的
try:
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# 常量定义
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 820
BOARD_MARGIN = 60
CELL_SIZE = 70
PIECE_RADIUS = 30

# 颜色
COLOR_BG = (245, 222, 179)       # 小麦色背景
COLOR_BOARD = (222, 184, 135)    # 棋盘颜色
COLOR_LINE = (139, 69, 19)      # 棋盘线颜色
COLOR_RED = (200, 0, 0)         # 红方
COLOR_BLACK = (20, 20, 20)      # 黑方
COLOR_SELECTED = (0, 200, 0)    # 选中高亮
COLOR_VALID_MOVE = (0, 150, 255) # 可移动位置
COLOR_LAST_MOVE = (255, 165, 0) # 上一步高亮
COLOR_BUTTON = (70, 130, 180)   # 按钮颜色
COLOR_BUTTON_HOVER = (100, 149, 237)

# 棋子类型
KING = 'k'     # 将/帅
ADVISOR = 'a'  # 士/仕
ELEPHANT = 'e' # 象/相
HORSE = 'h'    # 马
ROOK = 'r'     # 车
CANNON = 'c'   # 炮
PAWN = 'p'     # 兵/卒

# 方向
RED = 1
BLACK = -1

# 棋子基础分值（用于传统AI）
PIECE_VALUES = {
    KING: 10000,
    ADVISOR: 200,
    ELEPHANT: 200,
    HORSE: 400,
    ROOK: 900,
    CANNON: 450,
    PAWN: 100
}

PIECE_TO_CHANNEL = {
    (KING, RED): 0, (ADVISOR, RED): 1, (ELEPHANT, RED): 2,
    (HORSE, RED): 3, (ROOK, RED): 4, (CANNON, RED): 5, (PAWN, RED): 6,
    (KING, BLACK): 7, (ADVISOR, BLACK): 8, (ELEPHANT, BLACK): 9,
    (HORSE, BLACK): 10, (ROOK, BLACK): 11, (CANNON, BLACK): 12, (PAWN, BLACK): 13,
}

PIECE_NAMES = {
    (KING, RED): '帅', (KING, BLACK): '将',
    (ADVISOR, RED): '仕', (ADVISOR, BLACK): '士',
    (ELEPHANT, RED): '相', (ELEPHANT, BLACK): '象',
    (HORSE, RED): '马', (HORSE, BLACK): '马',
    (ROOK, RED): '车', (ROOK, BLACK): '车',
    (CANNON, RED): '炮', (CANNON, BLACK): '炮',
    (PAWN, RED): '兵', (PAWN, BLACK): '卒',
}


class ChineseChess:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]
        self.current_player = RED
        self.setup_initial_board()
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.check_warning = False

    def setup_initial_board(self):
        self.board[0][0] = (ROOK, BLACK)
        self.board[0][1] = (HORSE, BLACK)
        self.board[0][2] = (ELEPHANT, BLACK)
        self.board[0][3] = (ADVISOR, BLACK)
        self.board[0][4] = (KING, BLACK)
        self.board[0][5] = (ADVISOR, BLACK)
        self.board[0][6] = (ELEPHANT, BLACK)
        self.board[0][7] = (HORSE, BLACK)
        self.board[0][8] = (ROOK, BLACK)
        self.board[2][1] = (CANNON, BLACK)
        self.board[2][7] = (CANNON, BLACK)
        for i in range(0, 9, 2):
            self.board[3][i] = (PAWN, BLACK)

        self.board[9][0] = (ROOK, RED)
        self.board[9][1] = (HORSE, RED)
        self.board[9][2] = (ELEPHANT, RED)
        self.board[9][3] = (ADVISOR, RED)
        self.board[9][4] = (KING, RED)
        self.board[9][5] = (ADVISOR, RED)
        self.board[9][6] = (ELEPHANT, RED)
        self.board[9][7] = (HORSE, RED)
        self.board[9][8] = (ROOK, RED)
        self.board[7][1] = (CANNON, RED)
        self.board[7][7] = (CANNON, RED)
        for i in range(0, 9, 2):
            self.board[6][i] = (PAWN, RED)

    def clone(self):
        game = ChineseChess()
        game.board = copy.deepcopy(self.board)
        game.current_player = self.current_player
        game.game_over = self.game_over
        game.winner = self.winner
        game.move_history = copy.deepcopy(self.move_history)
        game.check_warning = self.check_warning
        return game

    def is_valid_position(self, row, col):
        return 0 <= row < 10 and 0 <= col < 9

    def in_palace(self, row, col, player):
        if player == RED:
            return 7 <= row <= 9 and 3 <= col <= 5
        else:
            return 0 <= row <= 2 and 3 <= col <= 5

    def is_friendly(self, row, col, player):
        piece = self.board[row][col]
        return piece is not None and piece[1] == player

    def is_enemy(self, row, col, player):
        piece = self.board[row][col]
        return piece is not None and piece[1] != player

    def get_valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        moves = []
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece[1] == player:
                    piece_type = piece[0]
                    if piece_type == KING:
                        moves.extend(self.get_king_moves(row, col, player))
                    elif piece_type == ADVISOR:
                        moves.extend(self.get_advisor_moves(row, col, player))
                    elif piece_type == ELEPHANT:
                        moves.extend(self.get_elephant_moves(row, col, player))
                    elif piece_type == HORSE:
                        moves.extend(self.get_horse_moves(row, col, player))
                    elif piece_type == ROOK:
                        moves.extend(self.get_rook_moves(row, col, player))
                    elif piece_type == CANNON:
                        moves.extend(self.get_cannon_moves(row, col, player))
                    elif piece_type == PAWN:
                        moves.extend(self.get_pawn_moves(row, col, player))
        return moves

    def get_king_moves(self, row, col, player):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.in_palace(new_row, new_col, player):
                if not self.is_friendly(new_row, new_col, player):
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_advisor_moves(self, row, col, player):
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.in_palace(new_row, new_col, player):
                if not self.is_friendly(new_row, new_col, player):
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_elephant_moves(self, row, col, player):
        moves = []
        directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            block_row, block_col = row + dr // 2, col + dc // 2
            if self.is_valid_position(new_row, new_col):
                if self.board[block_row][block_col] is None:
                    if not self.is_friendly(new_row, new_col, player):
                        if player == RED and new_row >= 5:
                            moves.append(((row, col), (new_row, new_col)))
                        elif player == BLACK and new_row <= 4:
                            moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_horse_moves(self, row, col, player):
        moves = []
        directions = [
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (1, -2, 0, -1),
            (-1, 2, 0, 1), (1, 2, 0, 1)
        ]
        for dr, dc, block_r, block_c in directions:
            new_row, new_col = row + dr, col + dc
            block_row, block_col = row + block_r, col + block_c
            if self.is_valid_position(new_row, new_col):
                if self.board[block_row][block_col] is None:
                    if not self.is_friendly(new_row, new_col, player):
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_rook_moves(self, row, col, player):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            for i in range(1, 10):
                new_row, new_col = row + dr * i, col + dc * i
                if not self.is_valid_position(new_row, new_col):
                    break
                if self.board[new_row][new_col] is None:
                    moves.append(((row, col), (new_row, new_col)))
                else:
                    if self.is_enemy(new_row, new_col, player):
                        moves.append(((row, col), (new_row, new_col)))
                    break
        return moves

    def get_cannon_moves(self, row, col, player):
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            jumped = False
            for i in range(1, 10):
                new_row, new_col = row + dr * i, col + dc * i
                if not self.is_valid_position(new_row, new_col):
                    break
                if not jumped:
                    if self.board[new_row][new_col] is None:
                        moves.append(((row, col), (new_row, new_col)))
                    else:
                        jumped = True
                else:
                    if self.board[new_row][new_col] is not None:
                        if self.is_enemy(new_row, new_col, player):
                            moves.append(((row, col), (new_row, new_col)))
                        break
        return moves

    def get_pawn_moves(self, row, col, player):
        moves = []
        direction = -1 if player == RED else 1
        new_row = row + direction
        if self.is_valid_position(new_row, col):
            if not self.is_friendly(new_row, col, player):
                moves.append(((row, col), (new_row, col)))
        if (player == RED and row <= 4) or (player == BLACK and row >= 5):
            for dc in [-1, 1]:
                new_col = col + dc
                if self.is_valid_position(row, new_col):
                    if not self.is_friendly(row, new_col, player):
                        moves.append(((row, col), (row, new_col)))
        return moves

    def make_move(self, move):
        (from_row, from_col), (to_row, to_col) = move
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        self.move_history.append((move, captured))

        if captured and captured[0] == KING:
            self.game_over = True
            self.winner = self.current_player

        self.check_warning = self.is_in_check(-self.current_player)
        self.current_player = -self.current_player

    def undo_move(self):
        if not self.move_history:
            return
        move, captured = self.move_history.pop()
        (from_row, from_col), (to_row, to_col) = move
        piece = self.board[to_row][to_col]
        self.board[from_row][from_col] = piece
        self.board[to_row][to_col] = captured
        self.current_player = -self.current_player
        self.game_over = False
        self.winner = None
        self.check_warning = False

    def is_in_check(self, player=None):
        if player is None:
            player = self.current_player
        king_pos = None
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece[0] == KING and piece[1] == player:
                    king_pos = (row, col)
                    break
        if not king_pos:
            return True

        opponent = -player
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece[1] == opponent:
                    piece_type = piece[0]
                    if piece_type == KING:
                        if row == king_pos[0]:
                            for c in range(min(col, king_pos[1]) + 1, max(col, king_pos[1])):
                                if self.board[row][c] is not None:
                                    break
                            else:
                                return True
                    elif piece_type == HORSE:
                        for dr, dc, block_r, block_c in [
                            (-2, -1, -1, 0), (-2, 1, -1, 0),
                            (2, -1, 1, 0), (2, 1, 1, 0),
                            (-1, -2, 0, -1), (1, -2, 0, -1),
                            (-1, 2, 0, 1), (1, 2, 0, 1)
                        ]:
                            if row + dr == king_pos[0] and col + dc == king_pos[1]:
                                if self.board[row + block_r][col + block_c] is None:
                                    return True
                    elif piece_type == ROOK:
                        if row == king_pos[0] or col == king_pos[1]:
                            clear = True
                            if row == king_pos[0]:
                                for c in range(min(col, king_pos[1]) + 1, max(col, king_pos[1])):
                                    if self.board[row][c] is not None:
                                        clear = False
                                        break
                            else:
                                for r in range(min(row, king_pos[0]) + 1, max(row, king_pos[0])):
                                    if self.board[r][col] is not None:
                                        clear = False
                                        break
                            if clear:
                                return True
                    elif piece_type == CANNON:
                        jumped = False
                        if row == king_pos[0]:
                            for c in range(min(col, king_pos[1]) + 1, max(col, king_pos[1])):
                                if self.board[row][c] is not None:
                                    if not jumped:
                                        jumped = True
                                    else:
                                        break
                            else:
                                if jumped:
                                    return True
                        elif col == king_pos[1]:
                            for r in range(min(row, king_pos[0]) + 1, max(row, king_pos[0])):
                                if self.board[r][col] is not None:
                                    if not jumped:
                                        jumped = True
                                    else:
                                        break
                            else:
                                if jumped:
                                    return True
                    elif piece_type == PAWN:
                        direction = -1 if opponent == RED else 1
                        if row + direction == king_pos[0] and col == king_pos[1]:
                            return True
                        if (opponent == RED and row <= 4) or (opponent == BLACK and row >= 5):
                            if row == king_pos[0] and abs(col - king_pos[1]) == 1:
                                return True
        return False

    def to_array(self, player=RED):
        board_array = np.zeros((14, 10, 9), dtype='float32')
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    channel = PIECE_TO_CHANNEL.get(piece, -1)
                    if channel >= 0:
                        board_array[channel, row, col] = 1.0
        if player == BLACK:
            board_array = board_array[:, ::-1, :]
        return board_array

    def move_to_index(self, move):
        (fr, fc), (tr, tc) = move
        return fr * 9 + fc, tr * 9 + tc

    def evaluate(self):
        score = 0
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    piece_type, player = piece
                    value = PIECE_VALUES[piece_type]
                    score += value * player
        return score


# 神经网络（仅当PaddlePaddle可用时）
if PADDLE_AVAILABLE:
    class ChessNet(nn.Layer):
        def __init__(self, hidden_size=256, num_blocks=3):
            super(ChessNet, self).__init__()
            self.hidden_size = hidden_size

            self.conv1 = nn.Conv2D(14, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2D(64)

            self.blocks = nn.LayerList([
                self._make_block(64) for _ in range(num_blocks)
            ])

            self.policy_conv = nn.Conv2D(64, 16, kernel_size=1)
            self.policy_bn = nn.BatchNorm2D(16)
            self.policy_fc = nn.Linear(16 * 10 * 9, 90 * 90)

            self.value_conv = nn.Conv2D(64, 16, kernel_size=1)
            self.value_bn = nn.BatchNorm2D(16)
            self.value_fc1 = nn.Linear(16 * 10 * 9, hidden_size)
            self.value_fc2 = nn.Linear(hidden_size, 1)

        def _make_block(self, channels):
            return nn.Sequential(
                nn.Conv2D(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2D(channels),
                nn.ReLU(),
                nn.Conv2D(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2D(channels),
            )

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))

            for block in self.blocks:
                residual = x
                x = F.relu(block(x) + residual)

            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = paddle.reshape(policy, [policy.shape[0], -1])
            policy = self.policy_fc(policy)

            value = F.relu(self.value_bn(self.value_conv(x)))
            value = paddle.reshape(value, [value.shape[0], -1])
            value = F.relu(self.value_fc1(value))
            value = paddle.tanh(self.value_fc2(value))

            return policy, value

        def predict_batch(self, boards):
            self.eval()
            with paddle.no_grad():
                x = paddle.to_tensor(np.array(boards))
                policy, value = self.forward(x)
                return policy.numpy(), value.numpy()


# MCTS Node（仅当PaddlePaddle可用时）
if PADDLE_AVAILABLE:
    class MCTSNode:
        def __init__(self, game, parent=None, move=None, prior_prob=0.0):
            self.game = game
            self.parent = parent
            self.move = move
            self.children = {}
            self.visit_count = 0
            self.total_value = 0.0
            self.prior_prob = prior_prob
            self.is_fully_expanded = False
            self.is_terminal = game.game_over


# MCTS（仅当PaddlePaddle可用时）
if PADDLE_AVAILABLE:
    class MCTS:
        def __init__(self, net, simulations=50, c_puct=1.0):
            self.net = net
            self.simulations = simulations
            self.c_puct = c_puct

        def get_action(self, game):
            root = MCTSNode(game.clone())
            legal_moves = game.get_valid_moves()

            if not legal_moves:
                return None

            board_array = game.to_array(game.current_player)
            policy, value = self.net.predict_batch([board_array])
            policy = policy[0]
            value = value[0][0]

            total_prior = 0
            for move in legal_moves:
                from_idx, to_idx = game.move_to_index(move)
                idx = from_idx * 90 + to_idx
                prior = np.exp(min(policy[idx], 10))
                total_prior += prior
                new_game = game.clone()
                new_game.make_move(move)
                root.children[move] = MCTSNode(new_game, parent=root, move=move, prior_prob=prior)

            for child in root.children.values():
                child.prior_prob /= (total_prior + 1e-8)

            root.is_fully_expanded = True

            for _ in range(self.simulations):
                self._run_simulation(root, value)

            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            return best_move

        def _run_simulation(self, root, root_value):
            node = root
            value = root_value

            while node.is_fully_expanded and not node.is_terminal:
                child = self._select_child(node)
                if child is None:
                    break
                node = child
                value = -value

            if not node.is_terminal and not node.is_fully_expanded:
                legal_moves = node.game.get_valid_moves()
                if legal_moves:
                    node.is_fully_expanded = True

            while node:
                node.visit_count += 1
                node.total_value += value
                value = -value
                node = node.parent

        def _select_child(self, node):
            if not node.children:
                return None
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                if child.visit_count == 0:
                    score = float('inf')
                else:
                    uct = self.c_puct * child.prior_prob * np.sqrt(node.visit_count) / (1 + child.visit_count)
                    score = child.total_value / child.visit_count + uct
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child


# AI Player with neural network
class AIPlayer:
    def __init__(self, model_path="chess_model_gpu.pdparams", use_nn=True):
        self.use_nn = False
        self.net = None
        self.mcts_simulations = 50

        if use_nn and PADDLE_AVAILABLE:
            try:
                self.net = ChessNet(hidden_size=256, num_blocks=3)
                if os.path.exists(model_path):
                    self.net.set_state_dict(paddle.load(model_path))
                    self.use_nn = True
                    print(f"已加载神经网络模型: {model_path}")
                else:
                    print(f"未找到模型文件 {model_path}，使用传统AI")
            except Exception as e:
                print(f"加载神经网络失败: {e}，使用传统AI")
        elif use_nn and not PADDLE_AVAILABLE:
            print("PaddlePaddle未安装，使用传统AI")

        if not self.use_nn:
            from functools import partial
            self.minimax_depth = 3

    def get_best_move(self, game, player=BLACK):
        if self.use_nn and self.net is not None and PADDLE_AVAILABLE:
            mcts = MCTS(self.net, simulations=self.mcts_simulations)
            return mcts.get_action(game)
        else:
            return self._minimax_best_move(game, player)

    def _minimax_best_move(self, game, player):
        best_move = None
        best_score = float('-inf') if player == RED else float('inf')

        moves = game.get_valid_moves(player)
        if not moves:
            return None

        moves.sort(key=lambda m: 1 if game.board[m[1][0]][m[1][1]] else 0, reverse=True)

        for move in moves:
            game.make_move(move)
            if game.is_in_check(-player):
                game.undo_move()
                continue

            score = self._minimax(game, self.minimax_depth - 1, float('-inf'), float('inf'), -player, player)
            game.undo_move()

            if player == RED:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def _minimax(self, game, depth, alpha, beta, player, maximizing_player):
        if depth == 0 or game.game_over:
            return game.evaluate()

        moves = game.get_valid_moves(player)
        if not moves:
            if game.is_in_check(player):
                return -100000 * player
            return 0

        if maximizing_player == RED:
            max_eval = float('-inf')
            for move in moves:
                game.make_move(move)
                eval_score = self._minimax(game, depth - 1, alpha, beta, -player, maximizing_player)
                game.undo_move()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                game.make_move(move)
                eval_score = self._minimax(game, depth - 1, alpha, beta, -player, maximizing_player)
                game.undo_move()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval


class GameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("中国象棋 - 人机对战")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('simhei', 48, bold=True)
        self.font_medium = pygame.font.SysFont('simhei', 32)
        self.font_small = pygame.font.SysFont('simhei', 24)

        self.game = ChineseChess()
        self.ai = AIPlayer(model_path="chess_model_gpu.pdparams", use_nn=True)
        self.human_player = RED
        self.ai_player = BLACK

        self.selected_piece = None
        self.valid_moves = []
        self.ai_thinking = False
        self.ai_thinking_time = 0
        self.show_start_screen = True
        self.show_game_over = False
        self.winner_text = ""
        self.ai_type = "神经网络" if self.ai.use_nn else "传统算法"

        self.btn_restart = pygame.Rect(WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT - 60, 160, 40)
        self.btn_new_game = pygame.Rect(WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT // 2 + 50, 160, 50)

    def board_to_screen(self, row, col):
        x = BOARD_MARGIN + col * CELL_SIZE
        y = BOARD_MARGIN + row * CELL_SIZE
        return x, y

    def screen_to_board(self, x, y):
        col = round((x - BOARD_MARGIN) / CELL_SIZE)
        row = round((y - BOARD_MARGIN) / CELL_SIZE)
        if 0 <= row < 10 and 0 <= col < 9:
            return row, col
        return None

    def draw_board(self):
        self.screen.fill(COLOR_BG)

        for i in range(10):
            x1 = BOARD_MARGIN + i * 0
            x2 = BOARD_MARGIN + 8 * CELL_SIZE
            y = BOARD_MARGIN + i * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE, (x1, y), (x2, y), 2)

        for i in range(9):
            x = BOARD_MARGIN + i * CELL_SIZE
            y1 = BOARD_MARGIN
            y2 = BOARD_MARGIN + 4 * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE, (x, y1), (x, y2), 2)

            y1 = BOARD_MARGIN + 5 * CELL_SIZE
            y2 = BOARD_MARGIN + 9 * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE, (x, y1), (x, y2), 2)

        palace_lines = [
            ((3, 0), (5, 2)),
            ((5, 0), (3, 2)),
            ((3, 7), (5, 9)),
            ((5, 7), (3, 9)),
        ]
        for (c1, r1), (c2, r2) in palace_lines:
            x1, y1 = self.board_to_screen(r1, c1)
            x2, y2 = self.board_to_screen(r2, c2)
            pygame.draw.line(self.screen, COLOR_LINE, (x1, y1), (x2, y2), 2)

        river_y = BOARD_MARGIN + 4.5 * CELL_SIZE
        text_chu = self.font_medium.render("楚河", True, COLOR_LINE)
        text_han = self.font_medium.render("汉界", True, COLOR_LINE)
        self.screen.blit(text_chu, (BOARD_MARGIN + 1 * CELL_SIZE, river_y - 15))
        self.screen.blit(text_han, (BOARD_MARGIN + 6 * CELL_SIZE, river_y - 15))

    def draw_pieces(self):
        if self.game.move_history:
            last_move, _ = self.game.move_history[-1]
            for pos in last_move:
                x, y = self.board_to_screen(*pos)
                pygame.draw.circle(self.screen, COLOR_LAST_MOVE, (x, y), PIECE_RADIUS + 5, 3)

        for move in self.valid_moves:
            x, y = self.board_to_screen(*move[1])
            target_piece = self.game.board[move[1][0]][move[1][1]]
            if target_piece:
                pygame.draw.circle(self.screen, (255, 100, 100), (x, y), PIECE_RADIUS + 3, 3)
            else:
                pygame.draw.circle(self.screen, COLOR_VALID_MOVE, (x, y), 8)

        for row in range(10):
            for col in range(9):
                piece = self.game.board[row][col]
                if piece:
                    x, y = self.board_to_screen(row, col)

                    if self.selected_piece == (row, col):
                        pygame.draw.circle(self.screen, COLOR_SELECTED, (x, y), PIECE_RADIUS + 5, 3)

                    pygame.draw.circle(self.screen, (250, 240, 230), (x, y), PIECE_RADIUS)
                    pygame.draw.circle(self.screen, (139, 69, 19), (x, y), PIECE_RADIUS, 2)

                    color = COLOR_RED if piece[1] == RED else COLOR_BLACK
                    text = self.font_large.render(PIECE_NAMES[piece], True, color)
                    text_rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, text_rect)

    def draw_ui(self):
        turn_text = "红方回合" if self.game.current_player == RED else "黑方回合"
        color = COLOR_RED if self.game.current_player == RED else COLOR_BLACK
        text = self.font_medium.render(turn_text, True, color)
        self.screen.blit(text, (20, 10))

        ai_type_text = self.font_small.render(f"AI: {self.ai_type}", True, (80, 80, 80))
        self.screen.blit(ai_type_text, (20, 45))

        if self.game.check_warning and not self.game.game_over:
            check_text = self.font_medium.render("将军!", True, (255, 0, 0))
            self.screen.blit(check_text, (WINDOW_WIDTH // 2 - 50, 10))

        if self.ai_thinking:
            ai_text = self.font_small.render(f"AI思考中... {self.ai_thinking_time:.1f}s", True, (100, 100, 100))
            self.screen.blit(ai_text, (WINDOW_WIDTH - 180, 20))

        btn_color = COLOR_BUTTON_HOVER if self.is_hovering(self.btn_restart) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.btn_restart, border_radius=10)
        btn_text = self.font_small.render("重新开始", True, (255, 255, 255))
        text_rect = btn_text.get_rect(center=self.btn_restart.center)
        self.screen.blit(btn_text, text_rect)

    def draw_start_screen(self):
        self.screen.fill(COLOR_BG)

        title = self.font_large.render("中国象棋", True, COLOR_LINE)
        subtitle = self.font_medium.render("人机对战", True, COLOR_LINE)

        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, WINDOW_HEIGHT // 3))
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, WINDOW_HEIGHT // 3 + 70))

        btn_color = COLOR_BUTTON_HOVER if self.is_hovering(self.btn_new_game) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.btn_new_game, border_radius=10)
        btn_text = self.font_medium.render("开始游戏", True, (255, 255, 255))
        text_rect = btn_text.get_rect(center=self.btn_new_game.center)
        self.screen.blit(btn_text, text_rect)

        info = [
            "操作说明:",
            "- 红方先手 (人类)",
            "- 点击选择棋子",
            "- 点击目标位置移动",
            "- 击败黑方 (AI) 获胜"
        ]
        for i, line in enumerate(info):
            text = self.font_small.render(line, True, (100, 100, 100))
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 + 150 + i * 30))

    def draw_game_over(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        winner_text = self.font_large.render(self.winner_text, True, (255, 215, 0))
        self.screen.blit(winner_text, (WINDOW_WIDTH // 2 - winner_text.get_width() // 2, WINDOW_HEIGHT // 3))

        btn_color = COLOR_BUTTON_HOVER if self.is_hovering(self.btn_new_game) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.btn_new_game, border_radius=10)
        btn_text = self.font_medium.render("再来一局", True, (255, 255, 255))
        text_rect = btn_text.get_rect(center=self.btn_new_game.center)
        self.screen.blit(btn_text, text_rect)

    def is_hovering(self, rect):
        mouse_pos = pygame.mouse.get_pos()
        return rect.collidepoint(mouse_pos)

    def handle_click(self, pos):
        if self.show_start_screen or self.show_game_over:
            if self.is_hovering(self.btn_new_game):
                self.game.reset()
                self.selected_piece = None
                self.valid_moves = []
                self.show_start_screen = False
                self.show_game_over = False
            return

        if self.is_hovering(self.btn_restart):
            self.game.reset()
            self.selected_piece = None
            self.valid_moves = []
            return

        if self.ai_thinking or self.game.game_over:
            return

        board_pos = self.screen_to_board(*pos)
        if not board_pos:
            return

        row, col = board_pos
        piece = self.game.board[row][col]

        for move in self.valid_moves:
            if move[1] == (row, col):
                self.game.make_move(move)
                self.selected_piece = None
                self.valid_moves = []
                return

        if piece and piece[1] == self.human_player:
            self.selected_piece = (row, col)
            moves = self.game.get_valid_moves(self.human_player)
            self.valid_moves = [m for m in moves if m[0] == (row, col)]
        else:
            self.selected_piece = None
            self.valid_moves = []

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)

            if self.show_start_screen:
                self.draw_start_screen()
            elif self.show_game_over:
                self.draw_board()
                self.draw_pieces()
                self.draw_game_over()
            else:
                self.draw_board()
                self.draw_pieces()
                self.draw_ui()

            pygame.display.flip()

            if (not self.show_start_screen and not self.show_game_over and
                not self.game.game_over and
                self.game.current_player == self.ai_player and
                not self.ai_thinking):

                self.ai_thinking = True
                start_time = time.time()

                best_move = self.ai.get_best_move(self.game, self.ai_player)
                self.ai_thinking_time = time.time() - start_time

                self.ai_thinking = False

                if best_move:
                    self.game.make_move(best_move)
                else:
                    self.game.game_over = True
                    self.game.winner = self.human_player
                    self.winner_text = "红方获胜!"
                    self.show_game_over = True

            if not self.show_start_screen and not self.show_game_over and self.game.game_over:
                if self.game.winner == RED:
                    self.winner_text = "红方获胜!"
                else:
                    self.winner_text = "黑方获胜!"
                self.show_game_over = True

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = GameUI()
    game.run()
