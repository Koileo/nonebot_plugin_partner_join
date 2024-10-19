from pydantic import BaseModel
import os
from nonebot import get_driver
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env.prod')

DEFAULT_PARAMS = {
    "skip_gif": ["-s", "s", "stop"],
    "self_join": ["自己", "我"]
}

DEFAULT_BACKGROUND_PARAMS = {
    "background.gif": ["default"],
}

DEFAULT_JOIN_COMMANDS = {"加入": ["旅行伙伴加入", "旋转"]}

params_str = os.getenv("PARAMS")
PARAMS = eval(params_str) if params_str else DEFAULT_PARAMS

background_params_str = os.getenv("BACKGROUND_PARAMS")
BACKGROUND_PARAMS = eval(background_params_str) if background_params_str else DEFAULT_BACKGROUND_PARAMS

commands_str = os.getenv("JOIN_COMMANDS")
JOIN_COMMANDS = eval(commands_str) if commands_str else DEFAULT_JOIN_COMMANDS

driver = get_driver()
driver.config.params = PARAMS
driver.config.background_params = BACKGROUND_PARAMS
driver.config.commands = JOIN_COMMANDS

class Config(BaseModel):
    """Plugin Config Here"""
