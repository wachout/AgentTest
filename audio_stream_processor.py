import socket
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# --- 配置参数 (请根据您的音频流进行修改) ---
HOST = '127.0.0.1'      # 服务器IP地址
PORT = 8888             # 服务器端口
CHUNK_DURATION = 10     # 每个音频块的处理持续时间 (秒)
OUTPUT_DIR = "output"   # 保存输出特征图的目录

# 音频流格式
SAMPLE_RATE = 44100     # 采样率 (Hz)
CHANNELS = 2            # 通道数
DTYPE = np.int16        # 数据类型 (例如: np.int16 -> 2 bytes per sample)

# 网络和数据处理
BUFFER_SIZE = 4096      # 每次从套接字接收的字节数
BYTES_PER_SAMPLE = DTYPE(0).itemsize
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
CHUNK_SIZE_BYTES = BYTES_PER_SECOND * CHUNK_DURATION


def process_audio_chunk(data_chunk, sr, channels, dtype):
    """
    处理原始字节数据块，提取指定通道并转换为 librosa 适用的浮点格式。
    """
    print("正在内存中直接处理音频数据块...")
    # 从缓冲区将原始字节转换为Numpy数组
    audio_array = np.frombuffer(data_chunk, dtype=dtype)

    # 如果是立体声，则重塑数组并提取第0通道
    if channels > 1:
        num_samples = len(audio_array) // channels
        audio_array = audio_array[:num_samples * channels]
        audio_array = audio_array.reshape((-1, channels))
        mono_audio = audio_array[:, 0]
    else:
        mono_audio = audio_array

    # 将音频数据转换为 librosa 喜欢的浮点格式 (-1.0 to 1.0)
    audio_float = mono_audio.astype(np.float32) / np.iinfo(dtype).max
    print("音频数据处理完成。")
    return audio_float

def extract_and_plot_features(audio_float, sr, output_image_path):
    """计算并绘制音频特征，然后将图像保存到文件。"""
    print("开始计算特征...")
    # 1. 计算声谱图
    stft = librosa.stft(audio_float)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # 2. 计算梅尔频谱图
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_float, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 3. 计算MFCC特征
    mfccs = librosa.feature.mfcc(y=audio_float, sr=sr, n_mfcc=40)

    print(f"特征计算完成，正在绘制并保存图像到: {output_image_path}")
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # 绘制声谱图
    img1 = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', ax=axs[0])
    axs[0].set_title('声谱图 (Spectrogram)')
    fig.colorbar(img1, ax=axs[0], format='%+2.0f dB')

    # 绘制梅尔频谱图
    img2 = librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title('梅尔频谱图 (Mel Spectrogram)')
    fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

    # 绘制MFCC
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axs[2])
    axs[2].set_title('MFCC 特征')
    fig.colorbar(img3, ax=axs[2])

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close(fig) # 关闭图形以释放内存
    print(f"成功保存特征图像: {output_image_path}")

def main():
    """主函数，持续接收音频流并处理。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出文件将保存在 '{OUTPUT_DIR}' 目录中。")

    data_buffer = b''

    try:
        print("脚本已启动，按 Ctrl+C 停止。")
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    print(f"正在尝试连接到 {HOST}:{PORT}...")
                    client_socket.connect((HOST, PORT))
                    print("连接成功！正在接收音频流...")

                    while True:
                        chunk = client_socket.recv(BUFFER_SIZE)
                        if not chunk:
                            print("服务器关闭了连接。正在尝试重新连接...")
                            break

                        data_buffer += chunk

                        if len(data_buffer) >= CHUNK_SIZE_BYTES:
                            print(f"\n已收集到足够的数据 ({len(data_buffer)} 字节)，准备处理一个 {CHUNK_DURATION} 秒的音频块。")

                            chunk_to_process = data_buffer[:CHUNK_SIZE_BYTES]
                            data_buffer = data_buffer[CHUNK_SIZE_BYTES:]

                            # 直接在内存中处理数据块
                            audio_float = process_audio_chunk(chunk_to_process, SAMPLE_RATE, CHANNELS, DTYPE)

                            # 生成文件名并进行特征提取和绘图
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            png_filename = f"features_{timestamp}.png"
                            png_filepath = os.path.join(OUTPUT_DIR, png_filename)
                            extract_and_plot_features(audio_float, SAMPLE_RATE, png_filepath)

            except ConnectionRefusedError:
                print(f"连接被拒绝。将在5秒后重试...")
                time.sleep(5)
            except Exception as e:
                print(f"发生未知错误: {e}。将在5秒后重试...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C！正在优雅地关闭程序...")
    finally:
        print("程序已停止。")

if __name__ == '__main__':
    main()
