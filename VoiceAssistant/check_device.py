import pyaudiowpatch as pyaudio

def main():
    p = pyaudio.PyAudio()
    try:
        # 获取 Windows WASAPI 宿主 API 信息
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("❌ 错误：未找到 WASAPI 驱动。")
        return

    print(f"\n{'ID':<5} | {'设备名称':<50} | {'输入通道':<8}")
    print("-" * 70)

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # 只显示属于 WASAPI 的设备
        if info["hostApi"] == wasapi_info["index"]:
            print(f"{info['index']:<5} | {info['name']:<50} | {info['maxInputChannels']:<8}")

    p.terminate()

if __name__ == "__main__":
    main()