import sys
import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QTextCursor

import config
from llm_engine import LLMEngine  # 即使是DeepSeek也建议沿用此入口名或自行修改


# --- 后台逻辑线程 ---
class VoiceWorker(QThread):
    user_signal = pyqtSignal(str)  # 用户转写结果
    ai_signal = pyqtSignal(str)  # AI 流式输出字符
    status_signal = pyqtSignal(str)  # 状态栏信息
    energy_signal = pyqtSignal(float)  # 音量能量值（可用于做动画）

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)
        self.qwen = LLMEngine()

    def run(self):
        p = pyaudio.PyAudio()
        self.status_signal.emit(f"🚀 环境就绪 ({self.device})")

        def get_stream(pa_obj):
            try:
                wasapi_info = pa_obj.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_info = pa_obj.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                loopback_device = None
                for i in range(pa_obj.get_device_count()):
                    info = pa_obj.get_device_info_by_index(i)
                    if (default_info["name"] in info["name"]) and (" [Loopback]" in info["name"]):
                        loopback_device = info
                        break
                if not loopback_device: return None, None

                s = pa_obj.open(
                    format=pyaudio.paInt16,
                    channels=loopback_device["maxInputChannels"],
                    rate=int(loopback_device["defaultSampleRate"]),
                    input=True,
                    input_device_index=loopback_device["index"]
                )
                return s, loopback_device
            except:
                return None, None

        stream, dev_info = get_stream(p)
        if not stream:
            self.status_signal.emit("❌ 未找到回环设备")
            return

        frames = []
        is_speaking = False
        silence_start = None

        while self.running:
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except:
                stream.close()
                stream, _ = get_stream(p)
                continue

            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_np ** 2))
            self.energy_signal.emit(energy)

            if energy > config.THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    frames = []
                    self.status_signal.emit("🎙️ 正在录音...")
                frames.append(data)
                silence_start = None
            else:
                if is_speaking:
                    if silence_start is None: silence_start = time.time()
                    if time.time() - silence_start > config.SILENCE_LIMIT:
                        self.status_signal.emit("📝 正在转写...")

                        # 保存临时音频
                        with wave.open(config.TEMP_AUDIO, 'wb') as wf:
                            wf.setnchannels(dev_info["maxInputChannels"])
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(int(dev_info["defaultSampleRate"]))
                            wf.writeframes(b''.join(frames))

                        # GPU 转写
                        res = self.asr_model.transcribe(
                            config.TEMP_AUDIO, language="zh",
                            fp16=(self.device == "cuda"), beam_size=5
                        )
                        user_text = res["text"].strip()

                        if user_text:
                            self.user_signal.emit(user_text)
                            self.status_signal.emit("🤖 助手正在思考...")
                            # 这里调用 LLM，注意：如果需要流式显示在UI，qwen_api需要微调
                            # 下面是简化演示，直接获取完整回复后再emit
                            reply = self.qwen.get_reply(user_text)
                            self.ai_signal.emit(reply)

                        is_speaking = False
                        silence_start = None
                        self.status_signal.emit("🟢 监听中...")

        stream.stop_stream()
        stream.close()
        p.terminate()


# --- 主界面 ---
class VoiceAssistantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 启动后台线程
        self.worker = VoiceWorker()
        self.worker.status_signal.connect(self.update_status)
        self.worker.user_signal.connect(self.add_user_chat)
        self.worker.ai_signal.connect(self.add_ai_chat)
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("AI 语音助手 - 毕设演示版")
        self.setFixedSize(600, 800)
        self.setStyleSheet("background-color: #F0F2F5;")

        layout = QVBoxLayout()

        # 状态栏
        self.status_bar = QLabel("正在初始化...")
        self.status_bar.setStyleSheet(
            "background-color: #FFFFFF; padding: 10px; border-radius: 5px; font-weight: bold;")
        layout.addWidget(self.status_bar)

        # 聊天显示区域
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("Microsoft YaHei", 11))
        self.chat_area.setStyleSheet("border: none; background-color: #FFFFFF; border-radius: 10px; padding: 15px;")
        layout.addWidget(self.chat_area)

        # 底部按钮区
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空对话")
        self.clear_btn.clicked.connect(lambda: self.chat_area.clear())
        self.clear_btn.setStyleSheet("background-color: #FF4D4F; color: white; padding: 8px; border-radius: 5px;")
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_status(self, msg):
        self.status_bar.setText(msg)

    def add_user_chat(self, text):
        self.chat_area.append(f"<div style='color: #1890FF;'><b>👤 我:</b><br>{text}</div><br>")
        self.chat_area.moveCursor(QTextCursor.MoveOperation.End)

    def add_ai_chat(self, text):
        self.chat_area.append(f"<div style='color: #52C41A;'><b>🤖 助手:</b><br>{text}</div><br>")
        self.chat_area.moveCursor(QTextCursor.MoveOperation.End)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceAssistantUI()
    window.show()
    sys.exit(app.exec())