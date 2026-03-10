import os
import time
import wave
import torch
import whisper
import numpy as np
import pyaudiowpatch as pyaudio

import config
from llm_engine import LLMEngine

# 初始化组件
# 只要你装好了带 +cu 的版本，这里会自动识别为 "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 正在加载 Whisper 模型 ({config.WHISPER_MODEL_SIZE})，设备: {device}...")

# 加载模型
asr_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=device)
qwen = LLMEngine()

def main():
    p = pyaudio.PyAudio()

    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("❌ 错误：未找到 WASAPI 驱动。")
        return

    # 寻找 Loopback 设备
    default_output_index = wasapi_info["defaultOutputDevice"]
    default_output_info = p.get_device_info_by_index(default_output_index)

    loopback_device = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if (default_output_info["name"] in info["name"]) and (" [Loopback]" in info["name"]):
            loopback_device = info
            break

    if not loopback_device:
        print("❌ 错误：未能找到可用的回环(Loopback)录音设备。")
        return

    print(f"\n🎯 监听中: {loopback_device['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=loopback_device["maxInputChannels"],
        rate=int(loopback_device["defaultSampleRate"]),
        input=True,
        input_device_index=loopback_device["index"]
    )

    print("🟢 系统文字助手已就绪...")

    frames = []
    is_speaking = False
    silence_start = None

    try:
        while True:
            data = stream.read(2048, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(audio_np ** 2))

            if energy > config.THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    frames = []
                    print("\n🎙️ 监测到声音...")
                frames.append(data)
                silence_start = None
            else:
                if is_speaking:
                    if silence_start is None: silence_start = time.time()
                    if time.time() - silence_start > config.SILENCE_LIMIT:
                        print("📝 正在精准转写...")

                        with wave.open(config.TEMP_AUDIO, 'wb') as wf:
                            wf.setnchannels(loopback_device["maxInputChannels"])
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(int(loopback_device["defaultSampleRate"]))
                            wf.writeframes(b''.join(frames))

                        # --- GPU 极致优化配置 ---
                        # 只要 device 是 cuda，fp16 必须设为 True 以释放 3050 的全部性能
                        use_fp16 = True if device == "cuda" else False

                        result = asr_model.transcribe(
                            config.TEMP_AUDIO,
                            language="zh",
                            fp16=use_fp16,  # 显卡加速的关键！
                            beam_size=5,    # 维持高精度
                            initial_prompt="这是一段关于毕业设计的对话，涉及到计算机技术和代码开发。"
                        )
                        user_text = result["text"].strip()

                        if user_text:
                            print(f"👤 用户: {user_text}")
                            qwen.get_reply(user_text)

                        is_speaking = False
                        silence_start = None
                        print("\n🟢 继续监听...")
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()