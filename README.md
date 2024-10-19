<div align="center">
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/NoneBotPlugin.svg" width="240" alt="NoneBotPluginText"></p>
</div>

<div align="center">
  
# nonebot-plugin-partner-join

_✨ NoneBot2 插件 用于生成舞萌DX(maimaiDX)旅行伙伴加入图片(旋转gif) 也可用于类似嵌入相应圆形框架图片生成(如将图片嵌入校徽)✨_

<a href="./LICENSE">
</a>
<img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="python">
</div>

## 📖 介绍

nonebot-plugin-partner-join 是一个基于本地数据的多功能机厅排卡报数插件，旨在为群聊舞萌玩家提供汇报机厅人数及线上排卡支持。该插件能够实现机厅人数上报，机厅排卡，添加机厅地图、别名，机厅状态管理和实时更新机厅人数状态等功能。
### 实现功能

- [x]  上报机厅人数
- [x]  显示当日更新过人数的机厅信息
- [x]  显示最新上报用户名及上报时间
- [x]  添加机厅别名
- [x]  显示群聊机厅别名列表
- [x]  添加机厅音游地图网址
- [x]  显示群聊机厅地图列表
- [x]  实现线上排卡功能

## 💿 安装

下载文件，将nonebot_plugin_partner_join文件夹放入您的nonebot2插件目录内

<details open>
<summary>使用 nb-cli 安装</summary> (暂不可用)
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-partner-join

</details>

<details>
<summary>使用包管理器安装</summary> (暂不可用)
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details>
<summary>pip</summary> (暂不可用)

    pip install nonebot-plugin-partner-join

</details>

打开 nonebot2 项目根目录下的 `pyproject.toml` 文件, 在 `[tool.nonebot]` 部分追加写入

    plugins = ["nonebot_plugin_partner_join"]

</details>

## ⚙️ 配置

在 nonebot2 项目的`.env`文件中添加下表中的必填配置

| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| JOIN_COMMANDS | 否 | {"加入": ["旅行伙伴加入", "旋转"]} | 加入指令，可自定义添加别名 |
| PARAMS | 否 | {"skip_gif": ["-s", "s", "stop"],"self_join": ["自己", "我"]} | 跳过生成旋转gif的参数及加入自己(头像图片)的指令 |
| BACKGROUND_PARAMS | 否 | {"background.gif": ["default"], "your_background_name.gif": ["指令1", "指令2"]} | 自定义将图片加入其他背景框的参数指令 |
| GIF_FPS | 否 | 30 | gif的fps |
| GIF_TOTAL_DURATION | 否 | 2 | gif的播放时间 |
| GIF_MAX_TURNS | 否 | 4 | gif的旋转圈数 |
| GIF_ROTATION_DIRECTION | 否 | -1 | gif的旋转方向(1 表示顺时针, -1 表示逆时针) |

## 🎉 使用
### 指令表
| 指令 | 权限 | 需要@ | 范围 | 说明 |
|:-----:|:----:|:----:|:----:|:----:|
| 加入/旅行伙伴加入 | 群员 | 否 | 群聊 | 生成"旅行伙伴加入"旋转gif\n(1.2.3.4.) |
| <机厅名>+num/-num | 群员 | 否 | 群聊 | +num/-num |
| <机厅名>=num/<机厅名>num| 群员 | 否 | 群聊 | 机厅的人数重置为num |
| <机厅名>几/几人/j | 群员 | 否 | 群聊 | 展示机厅当前的人数信息 |
| mai/机厅人数 | 群员 | 否 | 群聊 | 展示当日已更新的所有机厅的人数列表 |

### 效果图
(待传)
