import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import copy
import time
import os
import random

# 常量定义
KING = 'k'
ADVISOR = 'a'
ELEPHANT = 'e'
HORSE = 'h'
ROOK = 'r'
CANNON = 'c'
PAWN = 'p'

RED = 1
BLACK = -1

PIECE_TO_CHANNEL = {
    (KING, RED): 0, (ADVISOR, RED): 1, (ELEPHANT, RED): 2,
    (HORSE, RED): 3, (ROOK, RED): 4, (CANNON, RED): 5, (PAWN, RED): 6,
    (KING, BLACK): 7, (ADVISOR, BLACK): 8, (ELEPHANT, BLACK): 9,
    (HORSE, BLACK): 10, (ROOK, BLACK): 11, (CANNON, BLACK): 12, (PAWN, BLACK): 13,
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
        return game

    def is_valid_position(self, row, col):
        return 0 <= row < 10 and 0 <= col < 9

    def in_palace(self, row, col, player):
        if player == RED:
            return 7 <= row <= 9 and 3 <= col <= 5
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
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if self.in_palace(new_row, new_col, player):
                if not self.is_friendly(new_row, new_col, player):
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_advisor_moves(self, row, col, player):
        moves = []
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_row, new_col = row + dr, col + dc
            if self.in_palace(new_row, new_col, player):
                if not self.is_friendly(new_row, new_col, player):
                    moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_elephant_moves(self, row, col, player):
        moves = []
        for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
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
        for dr, dc, block_r, block_c in [
            (-2, -1, -1, 0), (-2, 1, -1, 0), (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (1, -2, 0, -1), (-1, 2, 0, 1), (1, 2, 0, 1)
        ]:
            new_row, new_col = row + dr, col + dc
            block_row, block_col = row + block_r, col + block_c
            if self.is_valid_position(new_row, new_col):
                if self.board[block_row][block_col] is None:
                    if not self.is_friendly(new_row, new_col, player):
                        moves.append(((row, col), (new_row, new_col)))
        return moves

    def get_rook_moves(self, row, col, player):
        moves = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
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
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
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
        self.current_player = -self.current_player

    def get_result(self, player=RED):
        if not self.game_over:
            return None
        return 1 if self.winner == player else -1

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


# GPU优化神经网络
class ChessNet(nn.Layer):
    def __init__(self, hidden_size=256, num_blocks=3):
        super(ChessNet, self).__init__()
        self.hidden_size = hidden_size

        # 卷积层
        self.conv1 = nn.Conv2D(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(64)

        # 残差块 (减少数量以节省内存)
        self.blocks = nn.LayerList([
            self._make_block(64) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2D(64, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2D(16)
        self.policy_fc = nn.Linear(16 * 10 * 9, 90 * 90)

        # Value head
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

        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = paddle.reshape(policy, [policy.shape[0], -1])
        policy = self.policy_fc(policy)

        # Value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = paddle.reshape(value, [value.shape[0], -1])
        value = F.relu(self.value_fc1(value))
        value = paddle.tanh(self.value_fc2(value))

        return policy, value

    def predict_batch(self, boards):
        """批量预测，提高GPU利用率"""
        self.eval()
        with paddle.no_grad():
            x = paddle.to_tensor(np.array(boards))
            policy, value = self.forward(x)
            return policy.numpy(), value.numpy()


# 简化的MCTS
class FastMCTS:
    def __init__(self, net, simulations=50, c_puct=1.0):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct

    def get_action(self, game):
        root = MCTSNode(game.clone())
        legal_moves = game.get_valid_moves()

        if not legal_moves:
            return None

        # 获取策略评估
        board_array = game.to_array(game.current_player)
        policy, value = self.net.predict_batch([board_array])
        policy = policy[0]
        value = value[0][0]

        # 初始化子节点
        total_prior = 0
        for move in legal_moves:
            from_idx, to_idx = game.move_to_index(move)
            idx = from_idx * 90 + to_idx
            prior = np.exp(min(policy[idx], 10))
            total_prior += prior
            new_game = game.clone()
            new_game.make_move(move)
            root.children[move] = MCTSNode(new_game, parent=root, move=move, prior_prob=prior)

        # 归一化
        for child in root.children.values():
            child.prior_prob /= (total_prior + 1e-8)

        root.is_fully_expanded = True

        # MCTS模拟
        for _ in range(self.simulations):
            self._run_simulation(root, value)

        # 选择最佳动作
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


class Trainer:
    def __init__(self, net=None, lr=0.001):
        if net is None:
            self.net = ChessNet(hidden_size=256, num_blocks=3)
        else:
            self.net = net

        self.optimizer = optim.Adam(parameters=self.net.parameters(), learning_rate=lr)
        self.training_data = []

    def self_play_batch(self, num_games=5, mcts_simulations=50):
        """批量自对弈"""
        all_data = []

        for game_idx in range(num_games):
            game = ChineseChess()
            mcts = FastMCTS(self.net, simulations=mcts_simulations)
            game_data = []
            move_count = 0

            while not game.game_over and move_count < 150:
                board_array = game.to_array(game.current_player)
                action = mcts.get_action(game)

                if action is None:
                    break

                policy = self._make_policy_target(action, game)
                game_data.append((board_array, policy))

                game.make_move(action)
                move_count += 1

            result = game.get_result(RED) if game.game_over else 0
            for i, (board, policy) in enumerate(game_data):
                value = result if i % 2 == 0 else -result
                all_data.append((board, policy, value))

            if (game_idx + 1) % 2 == 0:
                print(f"  完成 {game_idx + 1}/{num_games} 局")

        return all_data

    def _make_policy_target(self, action, game):
        probs = np.zeros(8100, dtype='float32')
        legal_moves = game.get_valid_moves()
        for move in legal_moves:
            from_idx, to_idx = game.move_to_index(move)
            idx = from_idx * 90 + to_idx
            probs[idx] = 1.0 if move == action else 0.1
        probs = probs / (np.sum(probs) + 1e-8)
        return probs

    def train(self, epochs=5, batch_size=64):
        """训练"""
        if len(self.training_data) < batch_size:
            print("数据不足，跳过训练")
            return

        for epoch in range(epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            num_batches = 0

            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i + batch_size]
                if len(batch) < batch_size // 2:
                    continue

                boards = np.array([d[0] for d in batch])
                policies = np.array([d[1] for d in batch])
                values = np.array([d[2] for d in batch], dtype='float32').reshape(-1, 1)

                boards = paddle.to_tensor(boards)
                policies = paddle.to_tensor(policies)
                values = paddle.to_tensor(values)

                pred_policies, pred_values = self.net(boards)

                policy_loss = F.cross_entropy(pred_policies, policies, soft_label=True)
                value_loss = F.mse_loss(pred_values, values)
                loss = policy_loss + value_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_gradients()

                total_loss += float(loss.numpy())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path="chess_model_gpu.pdparams"):
        try:
            paddle.save(self.net.state_dict(), path)
            print(f"模型已保存到 {path}")
        except Exception as e:
            print(f"保存模型失败: {e}")

    def load_model(self, path="chess_model_gpu.pdparams"):
        try:
            if os.path.exists(path):
                self.net.set_state_dict(paddle.load(path))
                print(f"模型已从 {path} 加载")
            else:
                print("使用新模型")
        except Exception as e:
            print(f"加载模型失败，使用新模型: {e}")

    def evaluate(self, games=5):
        wins, draws, losses = 0, 0, 0
        for _ in range(games):
            game = ChineseChess()
            mcts = FastMCTS(self.net, simulations=30)

            while not game.game_over:
                if game.current_player == RED:
                    action = mcts.get_action(game)
                else:
                    moves = game.get_valid_moves()
                    action = random.choice(moves) if moves else None

                if action:
                    game.make_move(action)
                else:
                    break

            result = game.get_result(RED)
            if result == 1: wins += 1
            elif result == -1: losses += 1
            else: draws += 1

        print(f"评估: {wins}胜 {draws}平 {losses}负")


def main():
    print("=== 中国象棋强化学习训练 (PaddlePaddle GPU优化版) ===")
    print(f"PaddlePaddle: {paddle.__version__}")
    print(f"CUDA可用: {paddle.is_compiled_with_cuda()}")
    print(f"GPU数量: {paddle.device.cuda.device_count()}")

    trainer = Trainer()
    trainer.load_model()

    # 优化参数 - 更小更稳定
    total_iterations = 20
    games_per_iteration = 5
    mcts_simulations = 50
    epochs_per_training = 5
    batch_size = 64

    for iteration in range(total_iterations):
        print(f"\n{'='*40}")
        print(f"迭代 {iteration + 1}/{total_iterations}")

        print("\n[自对弈]")
        start = time.time()
        new_data = trainer.self_play_batch(
            num_games=games_per_iteration,
            mcts_simulations=mcts_simulations
        )
        trainer.training_data.extend(new_data)
        print(f"生成 {len(new_data)} 条, 耗时 {time.time()-start:.1f}s")

        if len(trainer.training_data) > 10000:
            trainer.training_data = trainer.training_data[-10000:]

        print("\n[训练]")
        trainer.train(epochs=epochs_per_training, batch_size=batch_size)

        if (iteration + 1) % 5 == 0:
            print("\n[评估]")
            trainer.evaluate(games=5)
            trainer.save_model()

    trainer.save_model()
    print("\n训练完成!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
