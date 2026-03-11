from PyQt6.QtWidgets import (QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel,
                             QPushButton, QHBoxLayout, QLineEdit, QGroupBox,
                             QPlainTextEdit, QGridLayout)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QTextCursor
from modules.voice_worker import VoiceWorker


class VoiceAssistantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # 这里的名字决定了在电脑本地存储的位置
        self.settings = QSettings("MyGradProject", "JavaInterviewAssistant")
        self.init_ui()
        self.load_saved_config()
        self.worker = VoiceWorker()

        # 只有在已经填入 API Key 的情况下，才自动同步配置给引擎
        if self.key_input.text():
            self.save_and_apply()

        self.worker.status_signal.connect(self.update_status)
        self.worker.device_signal.connect(self.update_device_info)
        self.worker.user_signal.connect(self.add_user_chat)
        self.worker.ai_signal.connect(self.add_ai_chat)
        self.worker.start()

    def init_ui(self):
        self.setWindowTitle("Java面试智能助手")
        self.setFixedSize(550, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #FFFFFF; }
            QGroupBox { font-size: 12px; font-weight: bold; border: 1px solid #E8E8E8; border-radius: 8px; margin-top: 10px; }
            QTextEdit { background-color: #F7F8FA; border: none; border-radius: 10px; padding: 10px; }
            QPushButton#actionBtn { border-radius: 20px; color: white; font-weight: bold; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 状态栏
        header = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_label = QLabel(" 监听停止")
        self.device_label = QLabel("🎧 未连接")
        self.device_label.setStyleSheet("color: #999; font-size: 11px;")
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(self.device_label)
        layout.addLayout(header)

        # 配置区
        self.config_group = QGroupBox("系统设置")
        self.config_group.setCheckable(True)
        # 如果是第一次（配置为空），默认展开让用户填；如果已有配置，默认收起
        self.config_group.setChecked(True)

        grid = QGridLayout()
        self.url_input = QLineEdit()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_input = QLineEdit()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setFixedHeight(50)

        grid.addWidget(QLabel("Base URL:"), 0, 0);
        grid.addWidget(self.url_input, 0, 1)
        grid.addWidget(QLabel("API Key:"), 1, 0);
        grid.addWidget(self.key_input, 1, 1)
        grid.addWidget(QLabel("模型名称:"), 2, 0);
        grid.addWidget(self.model_input, 2, 1)
        grid.addWidget(QLabel("提示词:"), 3, 0);
        grid.addWidget(self.prompt_input, 3, 1)

        self.save_btn = QPushButton("💾 保存并应用配置")
        self.save_btn.clicked.connect(self.save_and_apply)
        grid.addWidget(self.save_btn, 4, 0, 1, 2)

        self.config_group.setLayout(grid)
        layout.addWidget(self.config_group)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        footer = QHBoxLayout()
        self.listen_btn = QPushButton("开始对话")
        self.listen_btn.setObjectName("actionBtn")
        self.listen_btn.setFixedSize(180, 40)
        self.update_btn_style(False)
        self.listen_btn.clicked.connect(self.toggle_listening)
        footer.addStretch();
        footer.addWidget(self.listen_btn);
        footer.addStretch()
        layout.addLayout(footer)


    def load_saved_config(self):
        """
        修改后的逻辑：默认值设为空字符串。
        只有用户在 UI 点击“保存”后，下次打开才会有值。
        """

        self.url_input.setText(self.settings.value("base_url", ""))
        self.key_input.setText(self.settings.value("api_key", ""))
        self.model_input.setText(self.settings.value("model_name", ""))
        self.prompt_input.setPlainText(self.settings.value("system_prompt", ""))

        # 如果已经有 API Key 了，就自动收起配置区，让界面更清爽
        if self.key_input.text():
            self.config_group.setChecked(False)

    def save_and_apply(self):
        url = self.url_input.text().strip()
        key = self.key_input.text().strip()
        model = self.model_input.text().strip()
        prompt = self.prompt_input.toPlainText().strip()

        if not key:
            self.add_sys_msg("⚠️ 请输入 API Key 后再保存")
            return

        # 持久化到本地存储
        self.settings.setValue("base_url", url)
        self.settings.setValue("api_key", key)
        self.settings.setValue("model_name", model)
        self.settings.setValue("system_prompt", prompt)

        # 同步给后端的 LLM 引擎
        if hasattr(self, 'worker'):
            self.worker.llm.update_config(key, url, model, prompt)
            self.add_sys_msg("✅ 配置已成功保存到本地")
            # 保存成功后自动收起
            self.config_group.setChecked(False)

    def toggle_listening(self):
        if not self.key_input.text():
            self.add_sys_msg("❌ 错误：请先配置 API 信息")
            self.config_group.setChecked(True)
            return
        new_state = not self.worker.is_listening
        self.worker.set_listening(new_state)
        self.update_btn_style(new_state)

    def update_btn_style(self, active):
        if active:
            self.listen_btn.setText("停止对话");
            self.listen_btn.setStyleSheet(
                "background-color: #FF4D4F; border-radius: 20px; color: white; font-weight: bold;")
            self.status_dot.setStyleSheet("color: #52C41A;");
            self.status_label.setText(" 监听中")
        else:
            self.listen_btn.setText("开始对话");
            self.listen_btn.setStyleSheet(
                "background-color: #1890FF; border-radius: 20px; color: white; font-weight: bold;")
            self.status_dot.setStyleSheet("color: #FF4D4F;");
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