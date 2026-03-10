import sys
import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QHBoxLayout,
                             QLineEdit, QFormLayout, QGroupBox, QPlainTextEdit,
                             QGridLayout, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QFont, QTextCursor, QColor

import config
from llm_engine import LLMEngine


# --- 1. 后台语音处理线程 ---
class VoiceWorker(QThread):
    user_signal = pyqtSignal(str)
    ai_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    device_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_listening = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载 Whisper 模型
        self.asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)
        self.llm = LLMEngine()

    def set_listening(self, state):
        self.is_listening = state

    def run(self):
        p = pyaudio.PyAudio()
        self.status_signal.emit(f"🚀 就绪 ({self.device})")

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
                return pa_obj.open(
                    format=pyaudio.paInt16,
                    channels=loopback_device["maxInputChannels"],
                    rate=int(loopback_device["defaultSampleRate"]),
                    input=True,
                    input_device_index=loopback_device["index"]
                ), loopback_device
            except:
                return None, None

        stream, dev_info = get_loopback_stream(p)
        if not stream:
            self.status_signal.emit("❌ 未找到音频设备")
            return
        self.device_signal.emit(dev_info['name'])

        frames = []
        is_speaking = False
        silence_start = None

        while self.running:
            if not self.is_listening:
                time.sleep(0.1)
                continue
            try:
                data = stream.read(2048, exception_on_overflow=False)
            except:
                stream.close()
                stream, _ = get_loopback_stream(p)
                continue

            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_np ** 2))

            if energy > config.THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    frames = []
                    self.status_signal.emit("🎙️ 录音中...")
                frames.append(data)
                silence_start = None
            else:
                if is_speaking:
                    if silence_start is None: silence_start = time.time()
                    if time.time() - silence_start > config.SILENCE_LIMIT:
                        self.status_signal.emit("📝 转写中...")
                        with wave.open(config.TEMP_AUDIO, 'wb') as wf:
                            wf.setnchannels(dev_info["maxInputChannels"])
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(int(dev_info["defaultSampleRate"]))
                            wf.writeframes(b''.join(frames))

                        result = self.asr_model.transcribe(config.TEMP_AUDIO, language="zh",
                                                           fp16=(self.device == "cuda"))
                        user_text = result["text"].strip()
                        if user_text:
                            self.user_signal.emit(user_text)
                            self.status_signal.emit("🤖 思考中...")
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
        self.settings = QSettings("MyGradProject", "JavaInterviewAssistant")
        self.init_ui()
        self.load_saved_config()

        self.worker = VoiceWorker()
        self.worker.status_signal.connect(self.update_status)
        self.worker.device_signal.connect(self.update_device_info)
        self.worker.user_signal.connect(self.add_user_chat)
        self.worker.ai_signal.connect(self.add_ai_chat)
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("Java面试智能语音助手")
        self.setFixedSize(550, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; }
            QGroupBox { font-size: 12px; font-weight: bold; color: #555; border: 1px solid #E8E8E8; border-radius: 8px; margin-top: 10px; padding: 10px; }
            QTextEdit { background-color: #F7F8FA; border: none; border-radius: 10px; padding: 10px; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QLineEdit, QPlainTextEdit { border: 1px solid #D9D9D9; border-radius: 4px; padding: 4px; }
            QPushButton#actionBtn { border-radius: 20px; color: white; font-weight: bold; font-size: 14px; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 15, 20, 15)

        # --- 顶部：状态 & 设备 ---
        header = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #FF4D4F; font-size: 14px;")
        self.status_label = QLabel(" 监听停止")
        self.device_label = QLabel("🎧 检测中...")
        self.device_label.setStyleSheet("color: #999; font-size: 11px;")
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(self.device_label)
        layout.addLayout(header)

        # --- 配置区：可勾选折叠 ---
        self.config_group = QGroupBox("系统设置 (API & 提示词)")
        self.config_group.setCheckable(True)
        self.config_group.setChecked(False)  # 默认收起

        grid = QGridLayout()
        self.url_input = QLineEdit()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_input = QLineEdit()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setFixedHeight(50)

        grid.addWidget(QLabel("Base URL:"), 0, 0)
        grid.addWidget(self.url_input, 0, 1)
        grid.addWidget(QLabel("API Key:"), 1, 0)
        grid.addWidget(self.key_input, 1, 1)
        grid.addWidget(QLabel("模型:"), 2, 0)
        grid.addWidget(self.model_input, 2, 1)
        grid.addWidget(QLabel("提示词:"), 3, 0)
        grid.addWidget(self.prompt_input, 3, 1)

        self.save_btn = QPushButton("💾 保存配置")
        self.save_btn.setStyleSheet("background-color: #FAFAFA; border: 1px solid #D9D9D9; padding: 5px;")
        self.save_btn.clicked.connect(self.save_and_apply)
        grid.addWidget(self.save_btn, 4, 0, 1, 2)

        self.config_group.setLayout(grid)
        layout.addWidget(self.config_group)

        # --- 聊天记录区 ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # --- 底部控制 ---
        footer = QHBoxLayout()
        self.listen_btn = QPushButton("开始对话")
        self.listen_btn.setObjectName("actionBtn")
        self.listen_btn.setFixedSize(180, 40)
        self.update_btn_style(False)
        self.listen_btn.clicked.connect(self.toggle_listening)

        self.clear_btn = QPushButton("🗑️ 清空")
        self.clear_btn.setFixedSize(60, 40)
        self.clear_btn.setStyleSheet("border: none; color: #999;")
        self.clear_btn.clicked.connect(lambda: self.chat_display.clear())

        footer.addStretch()
        footer.addWidget(self.listen_btn)
        footer.addStretch()
        footer.addWidget(self.clear_btn)
        layout.addLayout(footer)

    def load_saved_config(self):
        self.url_input.setText(self.settings.value("base_url", "https://api.deepseek.com"))
        self.key_input.setText(self.settings.value("api_key", ""))
        self.model_input.setText(self.settings.value("model_name", "deepseek-chat"))
        self.prompt_input.setPlainText(self.settings.value("system_prompt", "你是一个面试Java岗位的应届生，请简短回答。"))

    def save_and_apply(self):
        url, key, model, prompt = self.url_input.text(), self.key_input.text(), self.model_input.text(), self.prompt_input.toPlainText()
        self.settings.setValue("base_url", url)
        self.settings.setValue("api_key", key)
        self.settings.setValue("model_name", model)
        self.settings.setValue("system_prompt", prompt)
        self.worker.llm.update_config(key, url, model, prompt)
        self.add_sys_msg("系统配置已更新并应用")

    def toggle_listening(self):
        new_state = not self.worker.is_listening
        self.worker.set_listening(new_state)
        self.update_btn_style(new_state)

    def update_btn_style(self, active):
        if active:
            self.listen_btn.setText("结束对话")
            self.listen_btn.setStyleSheet(
                "background-color: #FF4D4F; border-radius: 20px; color: white; font-weight: bold;")
            self.status_dot.setStyleSheet("color: #52C41A; font-size: 14px;")
            self.status_label.setText(" 正在监听")
        else:
            self.listen_btn.setText("开始对话")
            self.listen_btn.setStyleSheet(
                "background-color: #1890FF; border-radius: 20px; color: white; font-weight: bold;")
            self.status_dot.setStyleSheet("color: #FF4D4F; font-size: 14px;")
            self.status_label.setText(" 监听停止")

    def update_status(self, msg):
        self.status_label.setText(f" {msg}")

    def update_device_info(self, name):
        self.device_label.setText(f"🎧 {name[:25]}")

    def add_user_chat(self, text):
        self.chat_display.insertHtml(
            f"<div style='margin: 8px 0;'><span style='background-color: #E6F7FF; padding: 6px 12px; border-radius: 8px;'><b>我:</b> {text}</span></div><br>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_ai_chat(self, text):
        self.chat_display.insertHtml(
            f"<div style='margin: 8px 0;'><span style='background-color: #F6FFED; padding: 6px 12px; border-radius: 8px;'><b>AI:</b> {text}</span></div><br>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_sys_msg(self, text):
        self.chat_display.append(f"<p style='color: #BFBFBF; font-size: 11px; text-align: center;'>— {text} —</p>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceAssistantUI()
    window.show()
    sys.exit(app.exec())