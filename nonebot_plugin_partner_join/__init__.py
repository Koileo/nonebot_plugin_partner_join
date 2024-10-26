import os
import time
import io
import re
import datetime
import httpx
from PIL import Image, ImageDraw, ImageSequence
from pathlib import Path
from nonebot.plugin import PluginMetadata
from nonebot import require, on_command, get_driver
from nonebot.adapters import Bot, Event, Message
from nonebot.adapters.onebot.v11 import Message, MessageSegment, Bot, Event, GroupMessageEvent
from nonebot.params import Arg, CommandArg, EventMessage
from nonebot.typing import T_State
require("nonebot_plugin_alconna")
from nonebot_plugin_alconna import Image as ALImage, UniMessage
from nonebot_plugin_alconna.uniseg.tools import image_fetch, reply_fetch
from nonebot_plugin_alconna.uniseg import UniMsg, Reply
require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler
require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store
from tarina import LRU
from typing import Optional
from nonebot.matcher import Matcher
from typing import List
from nonebot import get_plugin_config
from .config import Config
plugin_config = get_plugin_config(Config)

__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_partner_join",
    description="NoneBot2插件 用于生成舞萌DX(maimaiDX)旅行伙伴加入图片(旋转gif) 也可用于类似嵌入相应圆形框架图片生成(如将图片嵌入校徽)",
    usage="使用<加入帮助/join help>指令获取使用帮助",
    type="application",
    homepage="https://github.com/YuuzukiRin/nonebot_plugin_partner_join",
    config=Config,
    supported_adapters={"~onebot.v11"},
)

join_help = on_command("加入帮助", aliases={"join帮助", "加入help", "join help"}, priority=10, block=True)

@join_help.handle()
async def _(event: GroupMessageEvent, message: Message = EventMessage()):
    await join_help.send(
        "加入指令:\n"
        "[加入/join/旅行伙伴加入] 生成“旅行伙伴加入”旋转gif\n"
        "[加入+设置的加入其他背景框的指令] 换成你选择的背景框 如:加入XX\n"
        "指令参数:\n"
        "[<加入指令> -s/s/stop] 生成静态图片\n"
        "[<加入指令>我/me/自己] 加入自己(头像图片)\n"
        "指令使用:\n"
        "[<加入指令>image] 加入指令与图片一起发送\n"
        "[<加入指令>,image] 先发送加入指令再选择图片发送\n"
        "[<加入指令>“image”] 加入你引用的聊天记录(图片)\n"
        "[<加入指令>@XX] 加入@对象(头像图片)\n"
    ) 

join_DIR: Path = store.get_plugin_data_dir()
join_cache_DIR: Path = store.get_plugin_cache_dir()

@scheduler.scheduled_job('cron', hour=0, minute=0)
async def clear_join_daily():
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    if os.path.exists(join_DIR):
        for filename in os.listdir(join_DIR):
            file_path = os.path.join(join_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception:
                pass
                
PARAMS = plugin_config.params
SELF_PARAMS = plugin_config.self_params
BACKGROUND_PARAMS = plugin_config.background_params
JOIN_COMMANDS = plugin_config.join_commands

fps = plugin_config.gif_fps
total_duration = plugin_config.total_duration
max_turns = plugin_config.max_turns
rotation_direction = plugin_config.rotation_direction

async def extract_images(  
    bot: Bot, event: Event, state: T_State, msg: UniMsg
) -> str:  
    for msg_seg in msg:  
        if isinstance(msg_seg, ALImage): 

            return await image_fetch(bot=bot, event=event, state=state, img=msg_seg)

for main_command, aliases in JOIN_COMMANDS.items():
    join = on_command(main_command, aliases=set(aliases), priority=5, block=True)
    
@join.handle()
async def _(  
    bot: Bot,
    msg: UniMsg,
    event: Event,
    state: T_State,
    matcher: Matcher,
):

    for key in PARAMS.keys():
        state[key] = False

    for key, aliases in PARAMS.items():
        for alias in aliases:
            if any(alias in str(segment) for segment in msg): 
                state[key] = True
                break

    for key, aliases in SELF_PARAMS.items():
        for alias in aliases:
            if alias in str(msg) and not str(msg).lower().count("image"):
                state[key] = True
                break

    selected_background = "background.gif"
    for bg_file, aliases in BACKGROUND_PARAMS.items():
        for alias in aliases:
            if alias in str(msg):
                selected_background = bg_file
                break
    state["selected_background"] = selected_background 
    
    if msg.has(Reply):
        if (reply := await reply_fetch(event, bot)) and reply.msg:
            reply_msg = reply.msg            
            uni_msg_with_reply = UniMessage.generate_without_reply(message=reply_msg)
        msg.extend(uni_msg_with_reply)  
    
    if img_url := await extract_images(bot=bot, event=event, state=state, msg=msg):
        state["img_url"] = img_url  
        state["image_processed"] = True
    
    user_id = event.get_user_id()
    at_id = await plugin_config.get_at(event)    
    
    if at_id != "寄" and not state.get("image_processed", False):
        img_url = "url=https://q4.qlogo.cn/headimg_dl?dst_uin={}&spec=640,".format(at_id)
        state["image_processed"] = True
        state["image_object"] = True
    elif state.get("self_join", False):
        img_url = "url=https://q4.qlogo.cn/headimg_dl?dst_uin={}&spec=640,".format(user_id)
        state["image_processed"] = True
        state["image_object"] = True

    if state.get("image_object", False):
        url_pattern = re.compile(r'url=([^,]+)')
        match = url_pattern.search(img_url)
        if match:
            image_url = match.group(1)
            image_url = image_url.replace("&amp;", "&")
        else:
            print("未找到图片URL")

        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            img_data = response.content
            state["img_data"] = img_data
        
@join.got("image_processed", prompt="请选择要加入的旅行伙伴~(图片)")
async def handle_event(
    bot: Bot,
    msg: UniMsg,
    event: Event,
    state: T_State,
):
    if state.get("image_object", False):
        img_data = state["img_data"]
        await join.send("旅行伙伴加入中...")
        img = Image.open(io.BytesIO(img_data))
    else:
        img_data = await extract_images(bot=bot, event=event, state=state, msg=msg)
        if img_data:
            await join.send("旅行伙伴加入中...")
            img = Image.open(io.BytesIO(img_data))
        else:
            await join.finish("加入取消~")

    # 设置GIF路径
    gif_path = Path(join_cache_DIR) / "placeholder.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果跳过GIF
    if state.get("skip_gif", False):
        if getattr(img, "is_animated", False):
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=img.info.get("duration", 100))
        else:
            img = circle_crop(img)
            img.save(gif_path, format="GIF")
        state["skip_gif"] = False
    else:
        img = circle_crop(img)
        gif_path = Path(create_rotating_gif(img))

    # 合成背景和GIF图像
    background_path = Path(__file__).parent / "background" / state["selected_background"]
    final_gif_path = Path(composite_images(background_path, gif_path))

    if final_gif_path.exists():
        await join.send(MessageSegment.image(final_gif_path))
    else:
        print("生成的GIF图像文件不存在。")

    # 清理缓存的GIF文件
    if gif_path.exists():
        gif_path.unlink()

def circle_crop(img: Image.Image) -> Image.Image:
    """将图像裁剪成圆形，保留动态效果"""
    is_animated = getattr(img, "is_animated", False)
    if is_animated:
        frames = []
        for frame in ImageSequence.Iterator(img):
            cropped_frame = crop_single_frame(frame)
            frames.append(cropped_frame)

        output = frames[0]
        output.info = img.info

        gif_path = os.path.join(join_DIR, f"cropped_{int(time.time())}.gif")
        os.makedirs(join_DIR, exist_ok=True)
        output.save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=img.info.get("duration", 100))
        
        return Image.open(gif_path)
    else:
        return crop_single_frame(img)

def crop_single_frame(frame: Image.Image) -> Image.Image:
    """对单个帧进行圆形裁剪"""
    width, height = frame.size
    radius = min(width, height) // 2
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    center_x, center_y = width // 2, height // 2
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)
    output = Image.new("RGBA", (width, height))
    output.paste(frame, (0, 0), mask)
    output = output.crop((center_x - radius, center_y - radius, center_x + radius, center_y + radius))
    return output

def create_rotating_gif(img: Image.Image) -> str:
    """创建旋转GIF，保留动态效果"""
    frames = []
    num_frames = total_duration * fps
    max_angle = 360 * max_turns

    is_animated = getattr(img, "is_animated", False)
    original_frames = []

    if is_animated:
        original_frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

        original_num_frames = len(original_frames)
        if original_num_frames == num_frames:
            scaled_frames = original_frames
        elif original_num_frames < num_frames:
            # 如果原始帧数少于目标帧数，重复帧以填充
            repeat_count = (num_frames // original_num_frames) + 1
            scaled_frames = (original_frames * repeat_count)[:num_frames]
        else:
            # 如果原始帧数多于目标帧数，选择间隔帧进行等比缩放
            factor = original_num_frames / num_frames
            scaled_frames = [original_frames[int(i * factor)] for i in range(num_frames)]
    else:
        # 如果是静态图像，将静态图像处理为动态
        original_frames = [img] * num_frames
        scaled_frames = original_frames

    accel_duration = total_duration / 2  # 加速阶段和减速阶段时间相同
    accel_frames = accel_duration * fps
    decel_frames = accel_duration * fps
    total_frames = accel_frames + decel_frames

    # 计算加速阶段的角加速度
    accel_angle_change = 2 * max_angle / (accel_frames / fps) ** 2

    for i in range(num_frames):
        if i < accel_frames:
            # 加速阶段
            angle = 0.5 * accel_angle_change * (i / fps) ** 2
        else:
            # 减速阶段
            time_in_decel = i - accel_frames
            # 减速阶段角度计算
            angle = max_angle - 0.5 * accel_angle_change * ((accel_frames - time_in_decel) / fps) ** 2

        frame = scaled_frames[i].rotate(rotation_direction * angle, resample=Image.BICUBIC)
        frames.append(frame)

    output_dir = Path(join_DIR)
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"rotating_{int(time.time())}.gif"

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
    
    return str(gif_path)

def find_circle_diameter(mask: Image.Image) -> int:
    """计算掩码中圆形区域的直径"""
    width, height = mask.size
    center_x, center_y = width // 2, height // 2
    top_y = 0
    for y in range(center_y, -1, -1):
        if mask.getpixel((center_x, y)) > 0:
            top_y = y
            break
    bottom_y = height - 1
    for y in range(center_y, height):
        if mask.getpixel((center_x, y)) > 0:
            bottom_y = y
            break
    diameter = bottom_y - top_y + 1
    return diameter

def find_circle_center(mask: Image.Image) -> (int, int):
    """计算掩码中圆形区域的圆心"""
    width, height = mask.size
    center_x, center_y = width // 2, height // 2
    top_y = 0
    bottom_y = height - 1
    for y in range(center_y, -1, -1):
        if mask.getpixel((center_x, y)) > 0:
            top_y = y
            break
    for y in range(center_y, height):
        if mask.getpixel((center_x, y)) > 0:
            bottom_y = y
            break
    circle_center_y = top_y + (bottom_y - top_y) // 2
    return center_x, circle_center_y

def resize_gif_to_diameter(img: Image.Image, diameter: int) -> Image.Image:
    """将GIF图像等比缩放到指定的直径"""
    img = img.resize((diameter, diameter), Image.LANCZOS)
    return img

def composite_images(background_path: str, gif_path: str) -> str:
    """将GIF图像粘贴到背景图中"""
    background = Image.open(background_path).convert("RGBA")
    mask = background.split()[-1].convert("L")
    diameter = find_circle_diameter(mask)
    circle_center_x, circle_center_y = find_circle_center(mask)
    gif = Image.open(gif_path)

    gif_frames = []
    delays = []
    while True:
        try:
            frame = gif.copy()
            gif_frames.append(frame)
            delays.append(gif.info['duration'])
            gif.seek(gif.tell() + 1)
        except EOFError:
            break
    
    gif_frames = [circle_crop(frame) for frame in gif_frames]
    gif_frames = [resize_gif_to_diameter(frame, diameter) for frame in gif_frames]

    composite_frames = []
    for frame in gif_frames:
        composite_frame = background.copy()
        composite_frame.paste(frame, (circle_center_x - diameter // 2, circle_center_y - diameter // 2), frame.split()[-1])
        composite_frames.append(composite_frame)

    final_gif_path = Path(join_DIR) / f"composite_{int(time.time())}.gif"
    
    composite_frames[0].save(
        final_gif_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=delays,
        loop=0
    )
    
    return str(final_gif_path)
    
