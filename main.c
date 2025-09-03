#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// --- 配置参数 ---
#define SERVER_IP "127.0.0.1"
#define PORT 8888
#define SAMPLES_PER_FRAME 16384
#define CHANNELS_PER_SAMPLE 64
#define FRAME_SIZE (SAMPLES_PER_FRAME * CHANNELS_PER_SAMPLE * sizeof(uint32_t))

#define FRAME_DURATION_MS 170
#define BUFFER_DURATION_S 10
#define TARGET_CHANNEL 0

// --- 计算常量 ---
const int FRAMES_PER_10S = (int)((BUFFER_DURATION_S * 1000.0) / FRAME_DURATION_MS);
const int BUFFER_SAMPLES = SAMPLES_PER_FRAME * FRAMES_PER_10S;

// --- 特征结构体 (占位符) ---
typedef struct {
    float* mel_spectrogram;
    float* mfcc;
    int num_frames;
    int num_mfcc_coeffs;
} AudioFeatures;

// --- 函数声明 ---
void die(const char *message);
int create_socket();
void process_stream(int sock);
int receive_one_frame_and_extract(int sock, int16_t* target_buffer, uint8_t* frame_buffer);
AudioFeatures* calculate_features(const int16_t* audio_buffer, int num_samples);
int run_inference(const AudioFeatures* features);
void cleanup_features(AudioFeatures* features);

int main() {
    int sock = create_socket();
    process_stream(sock);
    close(sock);
    return 0;
}

void die(const char *message) {
    perror(message);
    exit(EXIT_FAILURE);
}

int create_socket() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        die("Socket creation failed");
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        die("Invalid address/ Address not supported");
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        die("Connection Failed");
    }
    printf("Connected to server %s:%d\n", SERVER_IP, PORT);
    return sock;
}

/**
 * @brief 主处理循环，持续接收数据，每10秒处理一次
 */
void process_stream(int sock) {
    // 为10秒的0通道数据分配缓冲区
    int16_t* audio_buffer_10s = (int16_t*)malloc(BUFFER_SAMPLES * sizeof(int16_t));
    if (!audio_buffer_10s) {
        die("Failed to allocate memory for 10s buffer");
    }

    // 为单帧原始数据分配缓冲区
    uint8_t* frame_buffer = (uint8_t*)malloc(FRAME_SIZE);
    if (!frame_buffer) {
        die("Failed to allocate memory for frame buffer");
    }

    printf("Starting real-time audio stream processing...\n");

    while (1) {
        printf("Collecting data for a new 10-second chunk...\n");
        for (int i = 0; i < FRAMES_PER_10S; ++i) {
            // 计算当前帧数据在10秒缓冲区中的偏移位置
            int16_t* buffer_offset = audio_buffer_10s + (i * SAMPLES_PER_FRAME);

            int status = receive_one_frame_and_extract(sock, buffer_offset, frame_buffer);

            if (status <= 0) { // 0表示连接关闭, -1表示错误
                if (status == 0) printf("Stream ended. Exiting.\n");
                free(audio_buffer_10s);
                free(frame_buffer);
                return; // 退出处理循环
            }
            printf("Received frame %d/%d\n", i + 1, FRAMES_PER_10S);
        }

        // 10秒数据已满，开始处理
        printf("\nBuffer full. Processing data...\n");
        printf("Calculating features...\n");
        AudioFeatures* features = calculate_features(audio_buffer_10s, BUFFER_SAMPLES);

        printf("Running inference...\n");
        int result = run_inference(features);
        printf("Inference result: %d\n\n", result);

        cleanup_features(features);
    }

    // 理论上不会执行到这里，除非while(1)中断
    free(audio_buffer_10s);
    free(frame_buffer);
}

/**
 * @brief 接收单帧数据，提取通道0，并存入目标缓冲区
 * @param sock 连接的套接字
 * @param target_buffer 指向10秒大缓冲区中当前帧应存入的位置
 * @param frame_buffer 用于接收原始帧数据的临时缓冲区
 * @return 1 表示成功, 0 表示连接关闭, -1 表示错误
 */
int receive_one_frame_and_extract(int sock, int16_t* target_buffer, uint8_t* frame_buffer) {
    ssize_t bytes_received = 0;
    while (bytes_received < FRAME_SIZE) {
        ssize_t result = recv(sock, (char*)frame_buffer + bytes_received, FRAME_SIZE - bytes_received, 0);
        if (result < 0) {
            perror("recv failed");
            return -1; // Error
        }
        if (result == 0) {
            printf("Connection closed by peer.\n");
            return 0; // Connection closed
        }
        bytes_received += result;
    }

    // 提取通道0的数据
    for (int j = 0; j < SAMPLES_PER_FRAME; ++j) {
        uint32_t* sample_ptr = (uint32_t*)(frame_buffer + j * CHANNELS_PER_SAMPLE * sizeof(uint32_t));
        uint32_t value_uint32 = *sample_ptr;
        target_buffer[j] = (int16_t)((value_uint32 >> 16) - 32768);
    }

    return 1; // Success
}

AudioFeatures* calculate_features(const int16_t* audio_buffer, int num_samples) {
    printf("  [STUB] Pretending to calculate Mel Spectrogram and MFCCs...\n");
    AudioFeatures* features = (AudioFeatures*)malloc(sizeof(AudioFeatures));
    if (!features) {
        die("Failed to allocate memory for features");
    }
    features->mel_spectrogram = NULL;
    features->mfcc = NULL;
    features->num_frames = 100;
    features->num_mfcc_coeffs = 13;
    return features;
}

int run_inference(const AudioFeatures* features) {
    printf("  [STUB] Pretending to run inference with the model...\n");
    return rand() % 100;
}

void cleanup_features(AudioFeatures* features) {
    if (features) {
        free(features);
    }
}
