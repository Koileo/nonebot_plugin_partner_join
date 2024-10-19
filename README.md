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

nonebot-plugin-partner-join 用于生成舞萌DX(maimaiDX)旅行伙伴加入图片(旋转gif)的插件 也可用于类似嵌入相应圆形框架图片生成(如将图片嵌入校徽)
### 实现原理

将用户发送的图片裁剪成圆形后嵌入背景gif的透明圆形区域
### 实现功能

- [x]  生成"旅行伙伴加入"旋转gif
- [x]  生成"旅行伙伴加入"静态图片
- [x]  自定义加入指令
- [x]  自定义命令参数
- [x]  自定义生成gif的各项参数
- [x]  自定义图片加入的背景框
- [x]  实现旋转gif的顺滑过渡
### 注意事项

如果需要增加自定义的背景图，请保证图片满足以下要求，将背景图放入background文件夹中，并在.env文件中按要求添加BACKGROUND_PARAMS配置
- 背景图片格式为.gif(可以直接通过修改文件名修改.png文件后缀为.gif)
- 背景图需要有圆形透明区域，需保证透明圆形区域圆心位于图片纵轴(不要求位于横轴)

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
| 加入/旅行伙伴加入 | 群员 | 否 | 群聊 | 生成"旅行伙伴加入"旋转gif(1.发送"加入+图片 "2.发送"加入"后再发送图片 3.引用图片并回复"加入" 4.发送"加入@你要加入的用户 5.发送"加入我/自己") |
| 加入/旅行伙伴加入 -s/s/stop | 群员 | 否 | 群聊 | 生成"旅行伙伴加入"静态图片 |
| 加入+设置的加入其他背景框的指令 | 群员 | 否 | 群聊 | 换成你选择的背景框 如:加入XX |

### 效果图
(待传)
