import os

# --- 核心配置 ---
DEEPSEEK_API_KEY = ""
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# --- 语音识别配置 ---
WHISPER_MODEL_SIZE = "small"
LANGUAGE = "zh"

# --- 声音触发配置 ---
THRESHOLD = 0.01
SILENCE_LIMIT = 1.5

# --- 缓存目录管理 (核心修改) ---
CACHE_DIR = "cache"
# 如果文件夹不存在，则自动创建
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# 动态拼接路径，确保文件生成在 cache 目录下
TEMP_AUDIO = os.path.join(CACHE_DIR, "temp_recording.wav")
RESPONSE_AUDIO = os.path.join(CACHE_DIR, "response.mp3")