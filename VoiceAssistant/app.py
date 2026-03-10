import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 语音助手 - 毕业设计项目")
        self.resize(500, 700)

        # 布局控件
        layout = QVBoxLayout()

        self.status_label = QLabel("🟢 系统就绪")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 14px; background-color: #f5f5f5;")

        layout.addWidget(self.status_label)
        layout.addWidget(self.chat_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 启动后台线程
        # self.worker = VoiceWorker()
        # self.worker.user_text_signal.connect(self.add_user_msg)
        # self.worker.ai_text_signal.connect(self.add_ai_msg)
        # self.worker.start()

    def add_user_msg(self, text):
        self.chat_display.append(f"<b>👤 用户:</b> {text}")

    def add_ai_msg(self, text):
        # 处理流式追加
        self.chat_display.insertPlainText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())