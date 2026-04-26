import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DeepSeekConfig:
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: int = 60
    max_retries: int = 3


@dataclass
class GameConfig:
    ai_player: str = "deepseek"
    ui_type: str = "pygame"
    first_player: str = "red"


def get_deepseek_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )


def get_game_config() -> GameConfig:
    return GameConfig(
        ai_player=os.getenv("AI_PLAYER", "deepseek"),
        ui_type=os.getenv("UI_TYPE", "pygame"),
        first_player=os.getenv("FIRST_PLAYER", "red"),
    )