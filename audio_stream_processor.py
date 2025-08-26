import socket
import numpy as np
import librosa
import time
from collections import deque

# --- 配置参数 (根据新的数据帧格式) ---
HOST = '127.0.0.1'      # 服务器IP地址
PORT = 8888             # 服务器端口

# --- 新的音频流格式定义 ---
SAMPLES_PER_FRAME = 16384
CHANNELS = 64
DTYPE_RECV = np.uint32    # 接收时的数据类型
FRAME_DURATION_S = 0.170  # 170ms

# --- 根据格式计算的参数 ---
SAMPLE_RATE = SAMPLES_PER_FRAME / FRAME_DURATION_S # ~96376.5 Hz
BYTES_PER_SAMPLE_PER_CHANNEL = DTYPE_RECV(0).itemsize # 4 bytes for uint32
FRAME_SIZE_BYTES = SAMPLES_PER_FRAME * CHANNELS * BYTES_PER_SAMPLE_PER_CHANNEL # 4,194,304 bytes

# --- 缓冲和处理参数 ---
BUFFER_DURATION_S = 10    # 缓冲10秒后进行处理
SAMPLES_PER_BUFFER = int(SAMPLE_RATE * BUFFER_DURATION_S)

# 网络缓冲区大小
BUFFER_SIZE = 4096 * 4 # 16KB

def receive_full_frame(sock, frame_size):
    """从套接字精确接收一个完整帧的数据。"""
    frame_buffer = bytearray(frame_size)
    view = memoryview(frame_buffer)
    bytes_received = 0
    while bytes_received < frame_size:
        remaining = frame_size - bytes_received
        chunk_size = min(BUFFER_SIZE, remaining)
        try:
            nbytes = sock.recv_into(view[bytes_received:], chunk_size)
            if nbytes == 0:
                return None # 连接已关闭
            bytes_received += nbytes
        except socket.timeout:
            print("Socket receive timed out.")
            continue
    return frame_buffer

def process_frame_to_int16(frame_data):
    """处理一帧的字节数据，提取通道0并转为int16。"""
    # 1. 解析为 uint32
    audio_array = np.frombuffer(frame_data, dtype=DTYPE_RECV)

    # 2. 重塑为 (SAMPLES_PER_FRAME, CHANNELS)
    try:
        audio_array = audio_array.reshape((SAMPLES_PER_FRAME, CHANNELS))
    except ValueError:
        print(f"错误: 无法重塑数组。接收到的数据大小可能不正确。")
        return None

    # 3. 提取第0通道
    channel_0_uint32 = audio_array[:, 0]

    # 4. 将 uint32 数据转换为 int16
    # 步骤 4a: uint32 -> float [-1.0, 1.0]
    float_data = (channel_0_uint32.astype(np.float64) - 2**31) / (2**31)
    # 步骤 4b: float [-1.0, 1.0] -> int16 [-32767, 32767]
    int16_data = (float_data * 32767).astype(np.int16)

    return int16_data

def calculate_features(audio_chunk_int16, sr):
    """
    接收一个 int16 的音频数据块，计算并返回各种特征。
    """
    print("接收到10秒数据，开始计算特征...")
    # 1. 将 int16 数据转换为 librosa 需要的 float 格式
    audio_float = audio_chunk_int16.astype(np.float32) / 32768.0

    # 2. 计算声谱图 (STFT)
    stft = librosa.stft(audio_float)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # 3. 计算梅尔频谱图
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_float, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 4. 计算MFCC特征
    mfccs = librosa.feature.mfcc(y=audio_float, sr=sr, n_mfcc=40)

    print("特征计算完成。")
    return stft_db, mel_spectrogram_db, mfccs

def main():
    """主函数，持续接收、缓冲、处理。"""
    print("--- 音频处理脚本启动 ---")
    print(f"期望帧大小: {FRAME_SIZE_BYTES / 1024**2:.2f} MB")
    print(f"计算采样率: {SAMPLE_RATE:.2f} Hz")
    print(f"每 {BUFFER_DURATION_S} 秒处理一次数据 (需要 {SAMPLES_PER_BUFFER} 个采样点)。")

    # 使用 deque 作为高效的缓冲区
    audio_buffer = deque()

    try:
        print("按 Ctrl+C 停止。")
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((HOST, PORT))
                    client_socket.settimeout(5.0) # 设置超时
                    print(f"成功连接到 {HOST}:{PORT}！等待数据帧...")

                    while True:
                        frame_data = receive_full_frame(client_socket, FRAME_SIZE_BYTES)

                        if not frame_data:
                            print("服务器关闭了连接。尝试重新连接...")
                            break

                        print(f"收到一帧数据 ({len(frame_data)} 字节)。")

                        # 处理帧，得到通道0的int16数据
                        int16_samples = process_frame_to_int16(frame_data)
                        if int16_samples is not None:
                            audio_buffer.extend(int16_samples)

                        # 检查缓冲区是否有足够的数据进行处理
                        if len(audio_buffer) >= SAMPLES_PER_BUFFER:
                            print(f"\n缓冲区已满 ({len(audio_buffer)} 个采样点)，准备处理 {BUFFER_DURATION_S} 秒的数据。")

                            # 从左侧取出所需数量的样本
                            samples_to_process = np.array([audio_buffer.popleft() for _ in range(SAMPLES_PER_BUFFER)])

                            # 计算特征
                            spectrogram, mel_spectrogram, mfccs = calculate_features(samples_to_process, SAMPLE_RATE)

                            # --- 特征已返回，可在此处进行后续计算 ---
                            print("演示：获取到的特征变量")
                            print(f"  - 声谱图 (Spectrogram) Shape: {spectrogram.shape}")
                            print(f"  - 梅尔频谱图 (Mel Spectrogram) Shape: {mel_spectrogram.shape}")
                            print(f"  - MFCCs Shape: {mfccs.shape}")
                            print("-----------------------------------------\n")
                            # 在这里，您可以添加您自己的代码来使用这些特征变量
                            # 例如: my_own_function(spectrogram, mel_spectrogram, mfccs)

            except ConnectionRefusedError:
                print(f"连接被拒绝。5秒后重试...")
                time.sleep(5)
            except socket.timeout:
                 print("连接超时。尝试重新连接...")
            except Exception as e:
                print(f"发生未知错误: {e}。5秒后重试...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C！正在关闭程序...")
    finally:
        print("程序已停止。")

if __name__ == '__main__':
    main()
