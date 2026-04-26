# Chinese AI Chess

中国象棋 AI 对弈游戏，通过 DeepSeek API 接入大模型进行智能对弈。

## 项目结构

```
ChineseAIChess/
├── config/
│   └── settings.py          # 配置文件
├── core/
│   ├── board.py             # 棋盘数据结构
│   └── move_validator.py    # 走棋规则验证
├── api/
│   ├── deepseek_client.py   # DeepSeek API 客户端
│   ├── prompt_builder.py     # Prompt 构建器
│   └── response_parser.py    # 响应解析
├── ui/
│   └── pygame_ui.py         # Pygame UI
├── game/
│   └── game_manager.py      # 游戏管理器
├── main.py                   # 程序入口
├── .env.example              # 环境变量模板
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 中设置你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=your_api_key_here
```

## 运行

```bash
python main.py
```

## 游戏说明

- 人类玩家执红方先走
- DeepSeek AI 执黑方
- 点击棋子选中，然后点击目标位置走棋
- 将死对方将/帅即获胜

## 依赖

- openai >= 1.0.0
- pygame >= 2.0.0
- python-dotenv >= 1.0.0