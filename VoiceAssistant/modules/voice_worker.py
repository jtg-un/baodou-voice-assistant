import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt6.QtCore import QThread, pyqtSignal
import config
from llm_engine import LLMEngine


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
        # 加载模型
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

                        # 1. 严格控制文件写入周期
                        with wave.open(config.TEMP_AUDIO, 'wb') as wf:
                            wf.setnchannels(dev_info["maxInputChannels"])
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(int(dev_info["defaultSampleRate"]))
                            wf.writeframes(b''.join(frames))

                        # 2. 关键：等待 0.1 秒，确保操作系统释放文件锁
                        time.sleep(0.1)

                        try:
                            # 3. 增强稳定性：fp16 设为 False，避免显卡驱动导致 FFmpeg 崩溃
                            result = self.asr_model.transcribe(
                                config.TEMP_AUDIO,
                                language="zh",
                                fp16=False
                            )

                            user_text = result["text"].strip()
                            if user_text:
                                self.user_signal.emit(user_text)
                                self.status_signal.emit("🤖 思考中...")
                                ai_reply = self.llm.get_reply(user_text)
                                self.ai_signal.emit(ai_reply)
                        except Exception as e:
                            print(f"ASR Error: {e}")
                            self.status_signal.emit("❌ 转写失败")

                        is_speaking = False
                        silence_start = None
                        self.status_signal.emit("🟢 监听中...")

        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()