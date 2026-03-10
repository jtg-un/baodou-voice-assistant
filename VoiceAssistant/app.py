import sys
import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QHBoxLayout,
                             QLineEdit, QFormLayout, QGroupBox, QPlainTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings
from PyQt6.QtGui import QFont, QTextCursor

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
        self.is_listening = False  # 初始状态：不监听
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载 Whisper 模型到显卡
        self.asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)
        self.llm = LLMEngine()

    def set_listening(self, state):
        """控制监听开关的方法"""
        self.is_listening = state

    def run(self):
        p = pyaudio.PyAudio()
        self.status_signal.emit(f"🚀 环境就绪 ({self.device})")

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
            self.status_signal.emit("❌ 错误：未找到音频流")
            return

        self.device_signal.emit(dev_info['name'])

        frames = []
        is_speaking = False
        silence_start = None

        while self.running:
            # 如果处于非监听状态，则跳过后续逻辑
            if not self.is_listening:
                time.sleep(0.1)  # 降低CPU占用
                continue

            try:
                data = stream.read(2048, exception_on_overflow=False)
            except OSError:
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
                        self.status_signal.emit("📝 正在转写...")

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
                            self.status_signal.emit("🤖 助手正在思考...")
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
        self.setWindowTitle("Java面试语音助手 - 毕设演示版")
        self.setFixedSize(650, 950)
        self.setStyleSheet("background-color: #F0F2F5;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 配置面板 ---
        config_group = QGroupBox("API & 模型配置")
        form_layout = QFormLayout()
        self.url_input = QLineEdit()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_input = QLineEdit()
        form_layout.addRow("Base URL:", self.url_input)
        form_layout.addRow("API Key:", self.key_input)
        form_layout.addRow("Model Name:", self.model_input)
        config_group.setLayout(form_layout)
        main_layout.addWidget(config_group)

        # --- 提示词面板 ---
        prompt_group = QGroupBox("系统提示词 (System Prompt)")
        prompt_layout = QVBoxLayout()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setFixedHeight(80)
        prompt_layout.addWidget(self.prompt_input)
        prompt_group.setLayout(prompt_layout)
        main_layout.addWidget(prompt_group)

        # --- 操作按钮 ---
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("💾 保存配置")
        self.apply_btn.setStyleSheet("background-color: #1890FF; color: white; padding: 10px; font-weight: bold;")
        self.apply_btn.clicked.connect(self.save_and_apply)

        # 开始/停止监听按钮
        self.listen_btn = QPushButton("▶ 开始监听")
        self.listen_btn.setStyleSheet("background-color: #52C41A; color: white; padding: 10px; font-weight: bold;")
        self.listen_btn.clicked.connect(self.toggle_listening)

        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.listen_btn)
        main_layout.addLayout(btn_layout)

        # --- 状态指示 ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("等待初始化...")
        self.device_label = QLabel("🎧 未连接")
        self.device_label.setStyleSheet("color: #999; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.device_label)
        main_layout.addLayout(status_layout)

        # --- 聊天记录 ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: white; border-radius: 8px; padding: 10px; font-size: 13px;")
        main_layout.addWidget(self.chat_display)

        self.clear_btn = QPushButton("清空聊天记录")
        self.clear_btn.clicked.connect(lambda: self.chat_display.clear())
        main_layout.addWidget(self.clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def load_saved_config(self):
        self.url_input.setText(self.settings.value("base_url", "https://api.deepseek.com"))
        self.key_input.setText(self.settings.value("api_key", ""))
        self.model_input.setText(self.settings.value("model_name", "deepseek-chat"))
        self.prompt_input.setPlainText(self.settings.value("system_prompt", "你是一个面试Java岗位的应届生。"))

    def save_and_apply(self):
        url = self.url_input.text().strip()
        key = self.key_input.text().strip()
        model = self.model_input.text().strip()
        prompt = self.prompt_input.toPlainText().strip()
        if not key:
            self.update_status("⚠️ 请填写 API Key")
            return
        self.settings.setValue("base_url", url)
        self.settings.setValue("api_key", key)
        self.settings.setValue("model_name", model)
        self.settings.setValue("system_prompt", prompt)
        self.worker.llm.update_config(key, url, model, prompt)
        self.add_system_info("✅ 配置已保存并应用到引擎")

    def toggle_listening(self):
        """切换监听状态的逻辑"""
        new_state = not self.worker.is_listening
        self.worker.set_listening(new_state)
        if new_state:
            self.listen_btn.setText("■ 停止监听")
            self.listen_btn.setStyleSheet("background-color: #FF4D4F; color: white; padding: 10px; font-weight: bold;")
            self.update_status("🟢 正在实时监听...")
        else:
            self.listen_btn.setText("▶ 开始监听")
            self.listen_btn.setStyleSheet("background-color: #52C41A; color: white; padding: 10px; font-weight: bold;")
            self.update_status("🛑 监听已暂停")

    def update_status(self, msg):
        self.status_label.setText(msg)

    def update_device_info(self, name):
        self.device_label.setText(f"🎧 {name[:25]}...")

    def add_user_chat(self, text):
        self.chat_display.append(f"<div style='color:#1890FF; margin-top:5px;'><b>👤 我:</b> {text}</div>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_ai_chat(self, text):
        self.chat_display.append(f"<div style='color:#52C41A; margin-top:5px;'><b>🤖 助手:</b> {text}</div><br>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def add_system_info(self, text):
        self.chat_display.append(f"<p style='color:gray; text-align:center; font-size:11px;'><i>{text}</i></p>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceAssistantUI()
    window.show()
    sys.exit(app.exec())