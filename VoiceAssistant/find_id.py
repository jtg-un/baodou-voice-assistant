import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)

print(f"\n{'ID':<5} | {'设备名称':<50} | {'输入通道':<8}")
print("-" * 70)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["hostApi"] == wasapi_info["index"]:
        # 重点关注名称里带 "Loopback" 且 "输入通道 > 0" 的设备
        print(f"{info['index']:<5} | {info['name']:<50} | {info['maxInputChannels']:<8}")

p.terminate()