# Java面试助手 (Java Interview Assistant) 🚀

一款基于 **OpenAI Whisper** 离线转写技术与 **大语言模型 (LLM)** 的实时面试辅助工具。通过 WASAPI 环回技术捕捉系统音频，实现面试官提问的即时转写与 AI 答案推导。

---

## ✨ 核心特性

- **🎧 系统音频环回采集**：利用 `PyAudioWPatch` 技术，直接抓取扬声器声音，无需外接麦克风即可识别面试官提问。
- **🎙️ 离线 ASR 转写**：内置 OpenAI Whisper 模型，支持毫秒级本地语音转文字，保护隐私且无需担心网络波动。
- **🤖 多模型智能对接**：深度兼容 OpenAI API 协议，支持 **火山方舟 (字节跳动/豆包)**、DeepSeek、GPT-4 等主流大模型。
- **⚡ 异步响应架构**：基于 **PyQt6** 的多线程 (`QThread`) 设计，转写与推理过程完全不卡顿界面。
- **📦 一键化部署**：提供 1.9GB 完整安装包，解压即用，免去复杂的 Python 环境配置。

---

## 🏗️ 技术架构



1. **音频层**: `PyAudioWPatch` 监听 Windows WASAPI 环回。
2. **处理层**: `Whisper` 执行语音序列到文本的端到端转换。
3. **推理层**: `LLM Engine` 调用方舟/OpenAI 接口生成回答。
4. **交互层**: `PyQt6` 负责双线程 UI 渲染与信号绑定。

---

## 🚀 快速开始

### 方式一：下载安装包 (推荐)
1. 前往 [Releases](此处替换为你的下载链接) 下载 `Java面试助手_安装程序.exe`。
2. 安装后需占用约 **5GB** 磁盘空间。
3. 运行前请确保开启 Windows **“立体声混音”**。

### 方式二：源码运行
```bash
# 克隆仓库
git clone [https://github.com/jtg-un/baodou-voice-assistant.git](https://github.com/jtg-un/baodou-voice-assistant.git)
cd baodou-voice-assistant

# 安装依赖
pip install -r requirements.txt

# 运行程序
python main.py
```
###
🛠️ 关键配置
1. 开启系统录音 (重要)
由于 Windows 安全限制，请手动开启：

右键音量图标 -> 声音 -> 录制。

右键空白处勾选 “显示禁用的设备”。

启用 “立体声混音 (Stereo Mix)”。

2. API 配置
在 config.py 中填写你的服务商信息：

BaseURL: 火山方舟通常为 https://ark.cn-beijing.volces.com/api/v3

API Key: 你的 API 密钥。

Model ID: 填入你在方舟后台创建的 推理接入点 ID。

📝 依赖环境 (Requirements)
Python 3.8+

PyQt6

openai-whisper

PyAudioWPatch

torch (CUDA recommended)

👨‍💻 作者
jtgun

项目作为娱乐使用。

📜 免责声明
本工具仅供技术交流与学术演示使用，请勿在正式面试中违反职业道德准则