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

# 棋子编码
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
        # 黑方
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

        # 红方
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

    def get_result(self, player=RED):
        if not self.game_over:
            return None
        return 1 if self.winner == player else -1

    def to_tensor(self, player=RED):
        # 转换为神经网络输入张量 [14, 10, 9]
        board_array = np.zeros((14, 10, 9), dtype='float32')
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    channel = PIECE_TO_CHANNEL.get(piece, -1)
                    if channel >= 0:
                        board_array[channel, row, col] = 1.0
        # 如果黑方视角，翻转棋盘
        if player == BLACK:
            board_array = board_array[:, ::-1, :]
        return paddle.to_tensor(board_array)

    def move_to_index(self, move):
        (fr, fc), (tr, tc) = move
        return fr * 9 + fc, tr * 9 + tc

    def index_to_move(self, from_idx, to_idx):
        fr, fc = from_idx // 9, from_idx % 9
        tr, tc = to_idx // 9, to_idx % 9
        return ((fr, fc), (tr, tc))


# 神经网络 (PaddlePaddle)
class ChessNet(nn.Layer):
    def __init__(self, hidden_size=256):
        super(ChessNet, self).__init__()
        self.hidden_size = hidden_size

        # 卷积层
        self.conv1 = nn.Conv2D(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(128, 256, kernel_size=3, padding=1)

        # 全连接层
        self.fc_policy = nn.Linear(256 * 10 * 9, hidden_size)
        self.fc_policy_out = nn.Linear(hidden_size, 90 * 90)  # 90*90 可能的移动

        self.fc_value = nn.Linear(256 * 10 * 9, hidden_size)
        self.fc_value_out = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape: [batch, 14, 10, 9]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = paddle.reshape(x, [x.shape[0], -1])

        # Policy head
        policy = F.relu(self.fc_policy(x))
        policy = self.dropout(policy)
        policy = self.fc_policy_out(policy)
        policy = paddle.reshape(policy, [-1, 90, 90])
        policy = paddle.reshape(policy, [-1, 8100])  # flatten

        # Value head
        value = F.relu(self.fc_value(x))
        value = self.dropout(value)
        value = paddle.tanh(self.fc_value_out(value))

        return policy, value

    def predict(self, board_tensor):
        self.eval()
        with paddle.no_grad():
            board_tensor = paddle.reshape(board_tensor, [1, 14, 10, 9])
            policy, value = self.forward(board_tensor)
            return paddle.reshape(policy, [8100]), value.numpy()[0][0]


# MCTS 节点
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


# 简化版 MCTS
class SimpleMCTS:
    def __init__(self, net, simulations=100, c_puct=1.0):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct

    def get_action(self, game):
        root = MCTSNode(game.clone())
        self._run(root)

        # 返回访问次数最多的动作
        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        return best_move

    def _run(self, root):
        for _ in range(self.simulations):
            node = root
            path = [node]

            # 选择
            while not node.is_terminal and node.is_fully_expanded:
                node = self._select_child(node)
                path.append(node)

            # 扩展和评估
            if not node.is_terminal:
                legal_moves = node.game.get_valid_moves()
                if legal_moves:
                    self._expand(node, legal_moves)
                    node = self._select_child(node)
                    path.append(node)

            # 模拟（使用神经网络评估）
            value = self._evaluate(node)

            # 回溯
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += value if n.game.current_player == RED else -value
                value = -value

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None

        for move, child in node.children.items():
            if child.visit_count == 0:
                score = float('inf')
            else:
                uct = self.c_puct * child.prior_prob * np.sqrt(node.visit_count) / (1 + child.visit_count)
                score = child.total_value / child.visit_count + uct

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node, legal_moves):
        # 获取神经网络预测
        board_tensor = node.game.to_tensor(node.game.current_player)
        policy, value = self.net.predict(board_tensor)

        # 创建子节点
        for move in legal_moves:
            from_idx, to_idx = node.game.move_to_index(move)
            idx = from_idx * 90 + to_idx
            prior = float(policy[idx])
            # 使用softmax归一化
            prior = np.exp(prior) if prior < 10 else np.exp(10)

            new_game = node.game.clone()
            new_game.make_move(move)

            node.children[move] = MCTSNode(new_game, parent=node, move=move, prior_prob=prior)

        node.is_fully_expanded = True

    def _evaluate(self, node):
        if node.game.game_over:
            return node.game.get_result(RED)

        # 使用神经网络评估
        board_tensor = node.game.to_tensor(RED)
        _, value = self.net.predict(board_tensor)
        return value


# 训练器
class Trainer:
    def __init__(self, net=None, lr=0.001):
        if net is None:
            self.net = ChessNet(hidden_size=256)
        else:
            self.net = net

        self.optimizer = optim.Adam(parameters=self.net.parameters(), learning_rate=lr)
        self.training_data = []

    def self_play(self, num_games=1, mcts_simulations=50):
        """自对弈生成训练数据"""
        all_data = []

        for game_num in range(num_games):
            game = ChineseChess()
            mcts = SimpleMCTS(self.net, simulations=mcts_simulations)

            game_data = []
            move_count = 0

            while not game.game_over and move_count < 200:
                board_tensor = game.to_tensor(game.current_player)
                action = mcts.get_action(game)

                if action is None:
                    break

                # 收集训练数据
                policy = self._get_action_probabilities(action, game)
                game_data.append((board_tensor, policy, 0))

                game.make_move(action)
                move_count += 1

            # 更新所有状态的value
            result = game.get_result(RED) if game.game_over else 0
            for i in range(len(game_data)):
                game_data[i] = (game_data[i][0], game_data[i][1], result if i % 2 == 0 else -result)

            all_data.extend(game_data)
            print(f"  自对弈 {game_num + 1}/{num_games} 完成，步数: {move_count}")

        return all_data

    def _get_action_probabilities(self, action, game):
        """获取动作的概率分布"""
        probs = np.zeros(8100, dtype='float32')
        legal_moves = game.get_valid_moves()
        for move in legal_moves:
            from_idx, to_idx = game.move_to_index(move)
            idx = from_idx * 90 + to_idx
            probs[idx] = 1.0 if move == action else 0.1
        # 归一化
        probs = probs / (np.sum(probs) + 1e-8)
        return paddle.to_tensor(probs)

    def train(self, epochs=10, batch_size=32):
        """训练神经网络"""
        if not self.training_data:
            print("没有训练数据，请先运行自对弈")
            return

        for epoch in range(epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            num_batches = 0

            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i + batch_size]
                if len(batch) < batch_size:
                    continue

                states = []
                target_policies = []
                target_values = []

                for state, policy, value in batch:
                    states.append(state.numpy() if isinstance(state, paddle.Tensor) else state)
                    target_policies.append(policy.numpy() if isinstance(policy, paddle.Tensor) else policy)
                    target_values.append(value)

                states = paddle.to_tensor(np.array(states))
                target_policies = paddle.to_tensor(np.array(target_policies))
                target_values = paddle.to_tensor(np.array(target_values, dtype='float32'))
                target_values = paddle.reshape(target_values, [-1, 1])

                # 前向传播
                predicted_policies, predicted_values = self.net(states)

                # 计算损失
                policy_loss = F.cross_entropy(predicted_policies, target_policies, soft_label=True)
                value_loss = F.mse_loss(predicted_values, target_values)
                loss = policy_loss + value_loss

                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_gradients()

                total_loss += float(loss.numpy())
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path="chess_model.pdparams"):
        paddle.save(self.net.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, path="chess_model.pdparams"):
        if os.path.exists(path):
            state_dict = paddle.load(path)
            self.net.set_state_dict(state_dict)
            print(f"模型已从 {path} 加载")
        else:
            print(f"模型文件 {path} 不存在，使用新模型")

    def evaluate(self, games=10):
        """评估模型与随机对手对弈"""
        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(games):
            game = ChineseChess()
            mcts = SimpleMCTS(self.net, simulations=30)

            while not game.game_over:
                if game.current_player == RED:
                    action = mcts.get_action(game)
                else:
                    legal_moves = game.get_valid_moves()
                    action = random.choice(legal_moves) if legal_moves else None

                if action:
                    game.make_move(action)
                else:
                    break

            result = game.get_result(RED)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        print(f"评估结果: {wins}胜 {draws}平 {losses}负 (vs 随机)")


def main():
    print("=== 中国象棋强化学习训练 (PaddlePaddle) ===")
    print(f"PaddlePaddle 版本: {paddle.__version__}")
    print(f"设备: {'GPU' if paddle.is_compiled_with_cuda() else 'CPU'}")
    print()

    # 创建训练器
    trainer = Trainer()
    trainer.load_model()

    # 训练参数
    total_iterations = 50
    games_per_iteration = 5
    mcts_simulations = 50
    epochs_per_training = 5

    for iteration in range(total_iterations):
        print(f"\n{'='*50}")
        print(f"迭代 {iteration + 1}/{total_iterations}")
        print(f"{'='*50}")

        # 自对弈
        print("\n[1/3] 生成训练数据...")
        start_time = time.time()
        new_data = trainer.self_play(num_games=games_per_iteration, mcts_simulations=mcts_simulations)
        trainer.training_data.extend(new_data)
        print(f"生成 {len(new_data)} 条数据，用时 {time.time() - start_time:.1f}秒")
        print(f"总训练数据: {len(trainer.training_data)} 条")

        # 限制数据量，避免内存溢出
        if len(trainer.training_data) > 10000:
            trainer.training_data = trainer.training_data[-10000:]

        # 训练
        print("\n[2/3] 训练神经网络...")
        trainer.train(epochs=epochs_per_training, batch_size=32)

        # 评估
        print("\n[3/3] 评估模型...")
        trainer.evaluate(games=5)

        # 保存模型
        if (iteration + 1) % 5 == 0:
            trainer.save_model()

    # 最终保存
    trainer.save_model()
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)


if __name__ == "__main__":
    main()
