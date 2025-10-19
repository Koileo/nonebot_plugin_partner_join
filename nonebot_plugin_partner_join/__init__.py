from nonebot import on_command
from nonebot.adapters.onebot.v11 import MessageSegment
import os
import time
import io
import re
import datetime
import httpx
from PIL import Image, ImageDraw, ImageSequence, ImageFilter
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
import os

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

    # 根据 static_image 动态选择参数
    if plugin_config.static_image:
        target_param = "rotate_img"
    else:
        target_param = "skip_gif"

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
            pass  # no image url found

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

    if plugin_config.static_image:
        # 当 static_image=True 时，rotate_img 参数控制是否生成 GIF
        should_generate_gif = state.get("rotate_img", False)
    else:
        # 当 static_image=False 时，skip_gif 参数控制是否跳过 GIF
        should_generate_gif = not state.get("skip_gif", False)

    if should_generate_gif:
        # 生成旋转 GIF
        img = circle_crop(img)
        gif_path = Path(create_rotating_gif(img))
    else:
        # 生成静态图
        if getattr(img, "is_animated", False):
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0,
                           duration=img.info.get("duration", 100))
        else:
            img = circle_crop(img)
            img.save(gif_path, format="GIF")
        state["skip_gif"] = False

    # 合成背景和GIF图像
    background_path = Path(__file__).parent / "background" / state["selected_background"]
    final_gif_path = Path(composite_images(background_path, gif_path))

    if final_gif_path.exists():
        await join.send(MessageSegment.image(final_gif_path))
    else:
        pass  # generated GIF not found

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

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)

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
        composite_frame.paste(frame, (circle_center_x - diameter // 2, circle_center_y - diameter // 2),
                              frame.split()[-1])
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


mirror = on_command("镜像", aliases={"对称", "mirror"}, priority=5, block=True)


def _ahash(img: Image.Image, hash_size: int = 8) -> str:
    g = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ''.join('1' if p >= avg else '0' for p in pixels)
    # convert to hex
    return hex(int(bits, 2))[2:].rjust(hash_size * hash_size // 4, '0')


def _dominant_colors(img: Image.Image, k: int = 5):
    # Use PIL quantize to approximate k dominant colors
    small = img.convert("RGB").resize((256, 256))
    pal = small.quantize(colors=k, method=Image.MEDIANCUT)
    counts = pal.getcolors(256 * 256) or []
    # Map palette index to RGB
    palette = pal.getpalette()
    out = []
    total = sum(c for c, _ in counts) or 1
    for c, idx in sorted(counts, reverse=True):
        r = palette[idx * 3 + 0];
        g = palette[idx * 3 + 1];
        b = palette[idx * 3 + 2]
        out.append((c / total, (r, g, b)))
    return out  # list of (ratio, (r,g,b))


def _edge_density(img: Image.Image) -> float:
    try:
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        pixels = list(edges.getdata())
        return sum(pixels) / (len(pixels) * 255.0)
    except Exception:
        return 0.0


@mirror.handle()
async def _(
        bot: Bot,
        msg: UniMsg,
        event: Event,
        state: T_State,
        matcher: Matcher,
        args=CommandArg()
):
    # 合并引用消息（与旅行伙伴一致）
    if msg.has(Reply):
        if (reply := await reply_fetch(event, bot)) and getattr(reply, "msg", None):
            uni_msg_with_reply = UniMessage.generate_without_reply(message=reply.msg)
            msg.extend(uni_msg_with_reply)

    # 优先从消息中取图
    if (img_url := await extract_images(bot=bot, event=event, state=state, msg=msg)):
        state["img_url"] = img_url
        state["image_processed"] = True

    # 尝试 @头像（与旅行伙伴一致）
    user_id = event.get_user_id()
    at_id = await plugin_config.get_at(event)
    if at_id != "寄" and not state.get("image_processed", False):
        img_url = "url=https://q4.qlogo.cn/headimg_dl?dst_uin={}&spec=640,".format(at_id)
        state["img_url"] = img_url
        state["image_processed"] = True
        state["image_object"] = True

    # 没有图片则进入 got 提示
    if not state.get("image_processed", False):
        await mirror.reject("请发送要对称的图片~(可直接发图，或引用一条带图消息)")


@mirror.got("image_processed")
async def _mirror_got(
        bot: Bot,
        msg: UniMsg,
        event: Event,
        state: T_State,
        matcher: Matcher,
):
    # 如果上一轮没取到，再尝试一次解析
    # 这里的 extract_images 会直接返回图片的二进制数据(bytes)
    if not state.get("img_url"):
        if (img_data := await extract_images(bot=bot, event=event, state=state, msg=msg)):
            state["img_url"] = img_data  # 将 bytes 存入 state

    if not state.get("img_url"):
        await mirror.finish("未检测到图片，对称处理取消~")

    # --- 以下是与 join 功能对齐的核心逻辑 ---

    img_bytes = None
    # 情况一：处理 @好友，需要下载头像URL
    if state.get("image_object", False):
        url_pattern = re.compile(r'url=([^,]+)')
        # state["img_url"] 此时是 "url=http://..." 格式的字符串
        match = url_pattern.search(state["img_url"])
        if match:
            image_url = match.group(1).replace("&amp;", "&")
        else:
            await mirror.finish("头像链接解析失败，请直接发送图片试试~")

        try:
            # 使用和 join 一样的简单下载方式
            async with httpx.AsyncClient() as client:
                resp = await client.get(image_url)
                resp.raise_for_status()  # 确保请求成功
            img_bytes = resp.content
        except Exception:
            await mirror.finish("头像下载失败，请稍后再试。")
    # 情况二：处理用户直接发送的图片
    else:
        # state["img_url"] 此时已经是 extract_images 返回的图片 bytes
        # 无需下载，直接使用即可！
        img_bytes = state["img_url"]

    # --- 图片处理逻辑保持不变，但增强了错误处理 ---

    # 解析参数（方向/基准侧/axis）——保留你之前的增强逻辑
    arg_text = ""
    try:
        from nonebot.params import CommandArg
    except Exception:
        pass
    direction = "horizontal"
    source_side = None
    axis_value = None

    # 解析图片并执行镜像
    try:
        # 确保 img_bytes 是有效数据后才进行处理
        if not img_bytes:
            await mirror.finish("未能获取到图片数据，处理取消~")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        w, h = img.size

        # 工具函数（与之前版本一致）
        def resolve_axis(is_horizontal: bool):
            if axis_value is None:
                return (w // 2) if is_horizontal else (h // 2)
            kind, val = axis_value
            if kind == "pct":
                if is_horizontal:
                    return max(0, min(w, round(w * val / 100.0)))
                else:
                    return max(0, min(h, round(h * val / 100.0)))
            else:
                return max(0, min((w if is_horizontal else h), val))

        def mirror_horizontal(src_side: str | None):
            x_axis = resolve_axis(True)
            left_box = (0, 0, x_axis, h)
            right_start = x_axis if w % 2 == 0 else x_axis + 1
            right_box = (right_start, 0, w, h)

            result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            if src_side in ("right",) and right_box[0] < right_box[2]:
                right = img.crop(right_box)
                right_mirror = right.transpose(Image.FLIP_LEFT_RIGHT)
                result.paste(right_mirror.crop((right_mirror.width - left_box[2], 0, right_mirror.width, h)), (0, 0))
                result.paste(img.crop((x_axis, 0, w, h)), (x_axis, 0))
            else:
                left = img.crop(left_box)
                left_mirror = left.transpose(Image.FLIP_LEFT_RIGHT)
                result.paste(img.crop((0, 0, x_axis, h)), (0, 0))
                result.paste(left_mirror.crop((0, 0, w - x_axis, h)), (x_axis, 0))
            return result

        def mirror_vertical(src_side: str | None, src_img: Image.Image | None = None):
            base = src_img if src_img is not None else img
            x_w, x_h = base.size
            y_axis = resolve_axis(False)
            top_box = (0, 0, x_w, y_axis)
            bottom_start = y_axis if x_h % 2 == 0 else y_axis + 1
            bottom_box = (0, bottom_start, x_w, x_h)

            result = Image.new("RGBA", (x_w, x_h), (0, 0, 0, 0))
            if src_side in ("bottom",):
                bottom = base.crop(bottom_box)
                bottom_mirror = bottom.transpose(Image.FLIP_TOP_BOTTOM)
                result.paste(bottom_mirror.crop((0, bottom_mirror.height - top_box[3], x_w, bottom_mirror.height)),
                             (0, 0))
                result.paste(base.crop((0, y_axis, x_w, x_h)), (0, y_axis))
            else:
                top = base.crop(top_box)
                top_mirror = top.transpose(Image.FLIP_TOP_BOTTOM)
                result.paste(base.crop((0, 0, x_w, y_axis)), (0, 0))
                result.paste(top_mirror.crop((0, 0, x_w, x_h - y_axis)), (0, y_axis))
            return result

        # 当前简单：保持默认（水平、左->右）
        if direction == "horizontal":
            result = mirror_horizontal("right" if source_side == "right" else "left")
        elif direction == "vertical":
            result = mirror_vertical("bottom" if source_side == "bottom" else "top")
        else:
            tmp = mirror_horizontal("right" if source_side == "right" else "left")
            result = mirror_vertical("bottom" if source_side == "bottom" else "top", tmp)

        # 保存与发送
        save_dir = Path("data");
        save_dir.mkdir(parents=True, exist_ok=True)
        if "A" in result.getbands() and result.getextrema()[-1][0] < 255:
            save_path = save_dir / f"mirror_{event.user_id}.png"
            result.save(save_path, format="PNG")
        else:
            result = result.convert("RGB")
            save_path = save_dir / f"mirror_{event.user_id}.jpg"
            result.save(save_path, format="JPEG", quality=95)

        await mirror.finish(MessageSegment.image(Path(save_path).resolve().as_uri()))
        try:
            os.remove(save_path)
        except Exception:
            pass
    except Exception:
        pass
