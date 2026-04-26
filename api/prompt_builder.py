from core.board import Board, Player, Move
from typing import List, Optional


class PromptBuilder:
    @staticmethod
    def get_system_prompt() -> str:
        return """你是一个专业的中国象棋 AI。你需要根据当前棋盘状态，选择一步最优走棋。

## 棋盘表示
- 棋盘为 10 行 x 9 列
- 行号 0-9：0 是黑方底线，9 是红方底线
- 列号 0-8：0 是左侧，8 是右侧
- 红方在棋盘下方（行 7-9），黑方在棋盘上方（行 0-2）

## 棋子编码
- 帅(k,RED)：将帅，在九宫内横竖走一步
- 士(a,RED)：仕士，在九宫内斜线走一步
- 象(e,RED)：相象，斜线走两格不过河，有象眼限制
- 马(h)：马，日字移动，有马腿限制
- 车(r)：车，横竖直线移动
- 炮(c)：炮，直线移动，吃子需要隔一个棋子（炮架）
- 兵(p,RED)：卒，过河前只能前进，过河后可左右移动

## 输出格式
你必须严格按以下格式输出：

思考过程：<你的推理过程>
走棋：<起点行>,<起点列> -> <终点行>,<终点列>
理由：<选择这步棋的原因>

## 重要规则
1. 你只能选择合法走棋
2. 不要送将（走出会导致自己被将军的棋）
3. 尽量争取胜利
4. 坐标使用 0-9（行）和 0-8（列）
5. 如果被将军，必须应将

请分析当前局面并给出你的走棋。"""

    @staticmethod
    def build(
        board: Board,
        player: Player,
        valid_moves: List[Move],
        move_history: Optional[List[str]] = None
    ) -> str:
        player_name = "红方" if player == Player.RED else "黑方"

        lines = [
            f"## 当前局面 - {player_name}回合",
            "",
            "### 棋盘状态",
            board.to_text_representation(player),
            "",
            "### 棋子位置说明",
            "符号格式：[颜色][棋子类型]",
            "- +k = 红帅, +a = 红仕, +e = 红相, +h = 红马, +r = 红车, +c = 红炮, +p = 红兵",
            "- -k = 黑将, -a = 黑士, -e = 黑象, -h = 黑马, -r = 黑车, -c = 黑炮, -p = 黑卒",
            "- 空位置用　表示",
            "",
        ]

        if move_history:
            lines.append("### 历史走棋")
            for i, move in enumerate(move_history[-10:], 1):
                lines.append(f"{i}. {move}")
            lines.append("")

        lines.extend([
            "### 合法走棋列表",
            "以下是所有可能的走棋（格式：起点->终点）：",
        ])

        for move in valid_moves[:30]:
            from_pos, to_pos = move
            lines.append(f"  {from_pos[0]},{from_pos[1]} -> {to_pos[0]},{to_pos[1]}")

        if len(valid_moves) > 30:
            lines.append(f"  ... 还有 {len(valid_moves) - 30} 种走棋")

        lines.append("")
        lines.append("请选择一步最优走棋。")

        return "\n".join(lines)