<div align="center">
  <!-- 
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="./docs/yuuzuki_join_s.gif" width="240" alt="NoneBotPluginText"></p>
  -->
  <p><img src="./docs/yuuzuki_join_s.gif" width="240" alt="NoneBotPluginText"></p>
  <a href="https://v2.nonebot.dev/store"><img src="./docs/NoneBotPlugin.svg" width="300" alt="logo"></a>
</div>

<div align="center">
  
# nonebot-plugin-partner-join

_✨ NoneBot2 插件 用于生成舞萌DX(maimaiDX)旅行伙伴加入图片(旋转gif) 也可用于类似嵌入相应圆形框架图片生成(如将图片嵌入校徽)✨_

<a href="./LICENSE">
</a>
<img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="python">
</div>

## 📖 介绍

nonebot-plugin-partner-join 是用于生成舞萌DX(maimaiDX)旅行伙伴加入图片(旋转gif)的插件 也可用于类似嵌入相应圆形框架图片生成(如将图片嵌入校徽)
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
- [x]  添加设置是否默认生成旋转gif
### 注意事项

如果需要增加自定义的背景图，请保证图片满足以下要求，将背景图放入background文件夹中，并在.env文件中按要求添加BACKGROUND_PARAMS配置
- 背景图片格式为.gif(可以直接通过修改文件名修改.png文件后缀为.gif)
- 背景图需要有圆形透明区域，需保证透明圆形区域圆心位于图片中轴线(纵向)

## 💿 安装
<details open>
<summary>使用包管理器安装</summary>
下载文件，将nonebot_plugin_partner_join文件夹放入您的nonebot2插件目录内(通常在 : 您的插件根目录\src\plugin)

</details>

<details>
<summary>使用 nb-cli 安装</summary> 
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-partner-join

</details>

<details open>
<summary>使用包管理器安装</summary> 
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details open>
<summary>pip</summary> 

    pip install nonebot-plugin-partner-join

</details>

打开 nonebot2 项目根目录下的 `pyproject.toml` 文件, 在 `[tool.nonebot]` 部分追加写入

    plugins = ["nonebot_plugin_partner_join"]

</details>

## ⚙️ 配置

在 nonebot2 项目的`.env`文件中添加下表中的必填配置

| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| JOIN_COMMANDS | 否 | {"加入": ["join", "旅行伙伴加入"]} | 加入指令，可自定义添加别名 |
| STATIC_IMAGE | 否 | True | False则默认生成旋转gif |
| PARAMS | 否 | {"skip_gif": ["-s", "s", "stop"], "rotate_img": ["-r", "r", "rotate"]} | 当 static_image=True 时，使用 rotate_img 参数生成gif；当 static_image=False 时，使用 skip_gif 参数跳过gif |
| SELF_PARAMS | 否 | {"self_join": ["自己", "me", "我"]} | 加入自己(头像图片)的指令 |
| BACKGROUND_PARAMS | 否 | {"background.gif": ["default"], "your_background_name.gif": ["指令1", "指令2"]} | 自定义将图片加入其他背景框的参数指令 |
| GIF_FPS | 否 | 30 | gif的fps |
| TOTAL_DURATION | 否 | 2 | gif的播放时间 |
| MAX_TURNS | 否 | 4 | gif的旋转圈数 |
| ROTATION_DIRECTION | 否 | -1 | gif的旋转方向(1 表示顺时针, -1 表示逆时针) |

## 🎉 使用
使用 `加入帮助/join help` 指令获取指令表
### 指令表
| 加入指令 | 范围 | 说明 |
|:-----:|:----:|:----:|
| 加入/join/旅行伙伴加入 | 群聊 | 生成"旅行伙伴加入"旋转gif |
| 加入+设置的加入其他背景框的指令 | 群聊 | 换成你选择的背景框 如:加入XX |

| 指令使用 | 范围 | 说明 |
|:-----:|:----:|:----:|
| <加入指令>image | 群聊 | 加入指令与图片一起发送 |
| <加入指令>,image | 群聊 | 先发送加入指令再选择图片发送 |
| <加入指令>"image" | 群聊 | 加入你引用的聊天记录(图片) |
| <加入指令>@XX | 群聊 | 加入@对象(头像图片) |

| 指令参数 | 范围 | 说明 |
|:-----:|:----:|:----:|
| -s/s/stop | 群聊 | 生成静态图片 |
| 我/me/自己 | 群聊 | 加入自己(头像图片) |

### 效果图
<details>
<summary>展开</summary> 

![image](https://github.com/YuuzukiRin/nonebot_plugin_partner_join/blob/main/docs/JOIN_COMMANDS_WITH.png)
![image](https://github.com/YuuzukiRin/nonebot_plugin_partner_join/blob/main/docs/JOIN_COMMANDS_SUBSTEP.png)
![image](https://github.com/YuuzukiRin/nonebot_plugin_partner_join/blob/main/docs/JOIN_COMMANDS_QUOTE.png)
![image](https://github.com/YuuzukiRin/nonebot_plugin_partner_join/blob/main/docs/JOIN_COMMANDS_AT.png)
![image](https://github.com/YuuzukiRin/nonebot_plugin_partner_join/blob/main/docs/JOIN_COMMANDS_ME.png)
