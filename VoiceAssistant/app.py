import sys
import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QHBoxLayout,
                             QLineEdit, QFormLayout, QGroupBox, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QTextCursor

import config
from llm_engine import LLMEngine  # 请确保你的文件名对应


# --- 1. 后台语音处理线程 ---
class VoiceWorker(QThread):
    user_signal = pyqtSignal(str)  # 传回用户转写文字
    ai_signal = pyqtSignal(str)  # 传回AI完整回复
    status_signal = pyqtSignal(str)  # 传回状态栏信息
    device_signal = pyqtSignal(str)  # 传回检测到的扬声器名称

    def __init__(self):
        super().__init__()
        self.running = True
        # 自动检测 CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)
        self.llm = LLMEngine()

    def run(self):
        p = pyaudio.PyAudio()
        self.status_signal.emit(f"🚀 环境就绪 (Device: {self.device})")

        def get_loopback_stream(pa_obj):
            try:
                wasapi_info = pa_obj.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_output = pa_obj.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                loopback_device = None
                for i in range(pa_obj.get_device_count()):
                    info = pa_obj.get_device_info_by_index(i)
                    if (default_output["name"] in info["name"]) and (" [Loopback]" in info["name"]):
                        loopback_device = info
                        break
                if not loopback_device: return None, None

                new_stream = pa_obj.open(
                    format=pyaudio.paInt16,
                    channels=loopback_device["maxInputChannels"],
                    rate=int(loopback_device["defaultSampleRate"]),
                    input=True,
                    input_device_index=loopback_device["index"]
                )
                return new_stream, loopback_device
            except:
                return None, None

        stream, dev_info = get_loopback_stream(p)
        if not stream:
            self.status_signal.emit("❌ 错误：未找到 WASAPI 回环设备")
            return

        self.device_signal.emit(dev_info['name'])

        frames = []
        is_speaking = False
        silence_start = None

        while self.running:
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError:
                # 自动重连逻辑
                stream.close()
                stream, _ = get_loopback_stream(p)
                continue

            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_np ** 2))

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
                        self.status_signal.emit("📝 正在精准转写...")

                        # 写入临时文件
                        with wave.open(config.TEMP_AUDIO, 'wb') as wf:
                            wf.setnchannels(dev_info["maxInputChannels"])
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(int(dev_info["defaultSampleRate"]))
                            wf.writeframes(b''.join(frames))

                        # 调用 RTX 3050 加速转写
                        result = self.asr_model.transcribe(
                            config.TEMP_AUDIO,
                            language="zh",
                            fp16=(self.device == "cuda")
                        )
                        user_text = result["text"].strip()

                        if user_text:
                            self.user_signal.emit(user_text)
                            self.status_signal.emit("🤖 助手正在思考...")
                            # 调用 LLM 接口
                            ai_reply = self.llm.get_reply(user_text)
                            self.ai_signal.emit(ai_reply)

                        is_speaking = False
                        silence_start = None
                        self.status_signal.emit("🟢 监听中...")

        stream.stop_stream()
        stream.close()
        p.terminate()


# --- 2. 主界面 UI ---
class VoiceAssistantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 启动后台工作线程
        self.worker = VoiceWorker()
        self.worker.status_signal.connect(self.update_status)
        self.worker.device_signal.connect(self.update_device_info)
        self.worker.user_signal.connect(self.add_user_chat)
        self.worker.ai_signal.connect(self.add_ai_chat)
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("Java面试语音助手 (RTX加速版)")
        self.setFixedSize(600, 850)
        self.setStyleSheet("background-color: #F8F9FA;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 配置面板 ---
        config_group = QGroupBox("API 自定义配置")
        config_group.setStyleSheet("font-weight: bold; color: #333;")
        form_layout = QFormLayout()

        self.url_input = QLineEdit("https://api.deepseek.com")
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("在此填入 API Key")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_input = QLineEdit("deepseek-chat")

        self.apply_btn = QPushButton("保存并应用配置")
        self.apply_btn.setStyleSheet("""
            QPushButton { background-color: #1890FF; color: white; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #40A9FF; }
        """)
        self.apply_btn.clicked.connect(self.save_config)

        form_layout.addRow("Base URL:", self.url_input)
        form_layout.addRow("API Key:", self.key_input)
        form_layout.addRow("Model Name:", self.model_input)
        form_layout.addRow(self.apply_btn)
        config_group.setLayout(form_layout)
        main_layout.addWidget(config_group)

        # --- 状态指示 ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("正在初始化系统...")
        self.device_label = QLabel("🎧 检测设备中...")
        self.device_label.setStyleSheet("color: #888; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.device_label)
        main_layout.addLayout(status_layout)

        # --- 聊天记录区 ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: white; 
            border: 1px solid #D9D9D9; 
            border-radius: 8px; 
            padding: 10px;
            font-family: 'Microsoft YaHei';
        """)
        main_layout.addWidget(self.chat_display)

        # --- 底部按钮 ---
        bottom_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清除屏幕")
        self.clear_btn.clicked.connect(lambda: self.chat_display.clear())
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.clear_btn)
        main_layout.addLayout(bottom_layout)

    def save_config(self):
        url = self.url_input.text().strip()
        key = self.key_input.text().strip()
        model = self.model_input.text().strip()

        if not key:
            self.update_status("⚠️ 请填写 API Key！")
            return

        # 同步配置到后台引擎
        self.worker.llm.update_config(key, url, model)
        self.add_system_info(f"系统配置已更新为: {model}")
        self.update_status("✅ 配置应用成功")

    def update_status(self, msg):
        self.status_label.setText(msg)

    def update_device_info(self, name):
        self.device_label.setText(f"🎧 监听中: {name[:25]}...")

    def add_user_chat(self, text):
        self.chat_display.append(f"<div style='margin-bottom:10px;'><b>👤 用户:</b><br>{text}</div>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_ai_chat(self, text):
        self.chat_display.append(f"<div style='margin-bottom:10px; color: #52C41A;'><b>🤖 助手:</b><br>{text}</div>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_system_info(self, text):
        self.chat_display.append(
            f"<div style='color: #888; font-size: 12px; text-align: center;'><i>--- {text} ---</i></div>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceAssistantUI()
    window.show()
    sys.exit(app.exec())