# Chinese AI Chess

中国象棋人工智能项目，支持 PaddlePaddle 和 PyTorch 双框架训练。

## 项目结构

```
ChineseAIChess/
├── chinese_chess_game.py       # 象棋游戏引擎与可视化
├── chinese_chess_paddle.py     # PaddlePaddle 训练
├── chinese_chess_paddle_gpu.py  # PaddlePaddle GPU 训练
├── chinese_chess_rl.py          # PyTorch 强化学习
└── chess_model_gpu.pdparams     # 预训练模型权重
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 游戏界面
```bash
python chinese_chess_game.py
```

### PaddlePaddle 训练
```bash
# CPU 训练
python chinese_chess_paddle.py

# GPU 训练
python chinese_chess_paddle_gpu.py
```

### PyTorch 强化学习
```bash
python chinese_chess_rl.py
```

## 依赖

- PaddlePaddle >= 3.0.0
- PyTorch >= 2.0.0
- PyGame >= 2.0.0
- NumPy >= 1.20.0
