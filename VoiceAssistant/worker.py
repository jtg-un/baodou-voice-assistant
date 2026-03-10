from PyQt6.QtCore import QThread, pyqtSignal
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
import time, wave, config
from llm_engine import LLMEngine  # 如果换了DeepSeek，记得改名


class VoiceWorker(QThread):
    # 定义信号：传回用户的话、助手的回复、以及当前状态
    user_text_signal = pyqtSignal(str)
    ai_text_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)
        self.llm = QwenEngine()

    def run(self):
        p = pyaudio.PyAudio()
        # ... (这里放你之前 main.py 里的 WASAPI 初始化代码) ...
        # 简化版逻辑示意：
        while self.running:
            # 1. 录音逻辑 (如 energy > threshold)
            # 2. 识别逻辑
            # self.status_signal.emit("正在转写...")
            # result = self.asr_model.transcribe(...)
            # self.user_text_signal.emit(result["text"])

            # 3. LLM 逻辑
            # self.llm.get_reply(text, callback=self.ai_text_signal.emit)
            pass