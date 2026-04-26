import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

from core.board import Board, Player, Move
from api.prompt_builder import PromptBuilder
from api.response_parser import ResponseParser


class DeepSeekAPIError(Exception):
    pass


class RateLimitError(DeepSeekAPIError):
    pass


class AuthenticationError(DeepSeekAPIError):
    pass


@dataclass
class AIAgentResponse:
    move: Optional[Move]
    reasoning: str
    is_valid: bool
    error_message: Optional[str] = None


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def get_ai_move(
        self,
        board: Board,
        player: Player,
        valid_moves: List[Move],
        move_history: Optional[List[str]] = None
    ) -> AIAgentResponse:
        prompt = PromptBuilder.build(board, player, valid_moves, move_history)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PromptBuilder.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                    timeout=self.timeout
                )

                content = response.choices[0].message.content

                move = ResponseParser.parse_move(content, valid_moves)

                return AIAgentResponse(
                    move=move,
                    reasoning=ResponseParser.extract_reasoning(content),
                    is_valid=move is not None
                )

            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or '429' in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    raise RateLimitError("API 速率限制")
                if 'auth' in error_str or '401' in error_str:
                    raise AuthenticationError("API 认证失败")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise DeepSeekAPIError(f"API 调用失败: {str(e)}")

        return AIAgentResponse(
            move=valid_moves[0] if valid_moves else None,
            reasoning="",
            is_valid=False,
            error_message="达到最大重试次数"
        )