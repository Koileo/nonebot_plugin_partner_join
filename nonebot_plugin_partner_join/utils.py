import nonebot
from nonebot.adapters.onebot.v11 import GroupMessageEvent

class Utils:
    def __init__(self) -> None:

        config = nonebot.get_driver().config 
        self.gif_fps: int = getattr(config, "GIF_FPS", 30)
        self.total_duration: int = getattr(config, "TOTAL_DURATION", 2)
        self.max_turns: int = getattr(config, "MAX_TURNS", 4)
        self.rotation_direction: int = getattr(config, "ROTATION_DIRECTION", -1)

    @staticmethod
    async def rule(event: GroupMessageEvent) -> bool:
        msg = event.get_message()
        return next(
            (msg_seg.data["qq"] != "all" for msg_seg in msg if msg_seg.type == "at"),
            False,
        )

    @staticmethod
    async def get_at(event: GroupMessageEvent) -> str:
        msg = event.get_message()
        return next(
            (
                "寄" if msg_seg.data["qq"] == "all" else str(msg_seg.data["qq"])
                for msg_seg in msg
                if msg_seg.type == "at"
            ),
            "寄",
        )

utils = Utils()
