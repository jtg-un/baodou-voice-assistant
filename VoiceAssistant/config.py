import os

# --- 核心配置 ---
# config.py
DEEPSEEK_API_KEY = "sk-d2223f9f90a04704bbdcf683d6ba2d60"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat" # 也就是 DeepSeek-V3

# --- 语音识别配置 ---
WHISPER_MODEL_SIZE = "base"  # 可选: tiny, base, small
LANGUAGE = "zh"

# --- 声音触发配置 ---
THRESHOLD = 0.01        # 灵敏度：数值越小越灵敏
SILENCE_LIMIT = 1.5     # 静音持续时间（秒）

# --- 临时文件路径 ---
TEMP_AUDIO = "temp_recording.wav"
RESPONSE_AUDIO = "response.mp3"