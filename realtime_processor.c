#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include "socket_client.h"
#include "mfcc.h"

// --- Configuration ---
#define SERVER_IP "192.168.35.6"
#define SERVER_PORT 8888
#define SAMPLES_PER_FRAME 16384
#define NUM_CHANNELS 64
#define FRAME_SIZE_BYTES (SAMPLES_PER_FRAME * NUM_CHANNELS * sizeof(uint32_t))
#define NUM_FRAMES_TO_ACCUMULATE 59
#define TOTAL_SAMPLES (NUM_FRAMES_TO_ACCUMULATE * SAMPLES_PER_FRAME)
#define SAMPLE_RATE 96000

// --- Data Structures ---
typedef struct {
    int num_frames;
    int num_spec_bins;
    int num_mels;
    int num_mfcc;
    double* power_spectrogram;
    double* mel_spectrogram;
    double* mfcc;
} AudioFeatures;

// --- Function Declarations ---
int receive_one_frame_and_extract(int sockfd, int16_t* out_buffer, uint32_t* raw_frame_buffer);
AudioFeatures* calculate_features(const int16_t* audio_buffer, int num_samples);
int run_inference(const AudioFeatures* features);
void cleanup_features(AudioFeatures* features);

// --- Function Implementations ---

int receive_one_frame_and_extract(int sockfd, int16_t* out_buffer, uint32_t* raw_frame_buffer) {
    if (socket_read_fully(sockfd, raw_frame_buffer, FRAME_SIZE_BYTES) != 0) {
        fprintf(stderr, "Failed to read full frame or connection closed.\n");
        return -1;
    }
    for (int i = 0; i < SAMPLES_PER_FRAME; ++i) {
        uint32_t sample_uint32 = raw_frame_buffer[i * NUM_CHANNELS];
        int32_t sample_int32 = (int32_t)(sample_uint32 - 2147483648U);
        out_buffer[i] = (int16_t)(sample_int32 >> 16);
    }
    return SAMPLES_PER_FRAME;
}

AudioFeatures* calculate_features(const int16_t* audio_buffer, int num_samples) {
    double* double_buffer = (double*)malloc(num_samples * sizeof(double));
    if (!double_buffer) {
        fprintf(stderr, "Failed to allocate memory for double buffer.\n");
        return NULL;
    }
    for (int i = 0; i < num_samples; ++i) {
        double_buffer[i] = (double)audio_buffer[i] / 32768.0;
    }

    mfcc_params params;
    params.sample_rate = SAMPLE_RATE;
    params.n_fft = 512;
    params.hop_length = 160;
    params.n_mels = 128;
    params.n_mfcc = 20;
    params.fmin = 0.0;
    params.fmax = SAMPLE_RATE / 2.0;

    int num_frames = 1 + (num_samples - params.n_fft) / params.hop_length;
    unsigned int spec_len = next_power_of_2(params.n_fft) / 2 + 1;

    AudioFeatures* features = (AudioFeatures*)malloc(sizeof(AudioFeatures));
    if (!features) {
        fprintf(stderr, "Failed to allocate memory for features struct.\n");
        free(double_buffer);
        return NULL;
    }
    features->power_spectrogram = (double*)malloc(num_frames * spec_len * sizeof(double));
    features->mel_spectrogram = (double*)malloc(num_frames * params.n_mels * sizeof(double));
    features->mfcc = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));

    if (!features->power_spectrogram || !features->mel_spectrogram || !features->mfcc) {
        fprintf(stderr, "Failed to allocate memory for feature buffers.\n");
        cleanup_features(features);
        free(double_buffer);
        return NULL;
    }

    features->num_frames = num_frames;
    features->num_spec_bins = spec_len;
    features->num_mels = params.n_mels;
    features->num_mfcc = params.n_mfcc;

    int frames_computed = compute_features(&params, double_buffer, num_samples,
                                           features->power_spectrogram,
                                           features->mel_spectrogram,
                                           features->mfcc);
    free(double_buffer);

    if (frames_computed <= 0) {
        fprintf(stderr, "Feature computation failed.\n");
        cleanup_features(features);
        return NULL;
    }

    return features;
}

int run_inference(const AudioFeatures* features) {
    if (!features) return -1;
    printf("--- Inference Simulation ---\n");
    printf("Successfully received features for %d frames.\n", features->num_frames);
    printf("Power Spectrogram (Frame 0, first 5): [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
           features->power_spectrogram[0], features->power_spectrogram[1], features->power_spectrogram[2], features->power_spectrogram[3], features->power_spectrogram[4]);
    printf("Mel Spectrogram (Frame 0, first 5):   [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
           features->mel_spectrogram[0], features->mel_spectrogram[1], features->mel_spectrogram[2], features->mel_spectrogram[3], features->mel_spectrogram[4]);
    printf("MFCC (Frame 0, first 5):            [%.2f, %.2f, %.2f, %.2f, %.2f...]\n",
           features->mfcc[0], features->mfcc[1], features->mfcc[2], features->mfcc[3], features->mfcc[4]);
    return 0;
}

void cleanup_features(AudioFeatures* features) {
    if (features) {
        if (features->power_spectrogram) free(features->power_spectrogram);
        if (features->mel_spectrogram) free(features->mel_spectrogram);
        if (features->mfcc) free(features->mfcc);
        free(features);
    }
}

void process_stream() {
    // Allocate buffers that will be reused in the loop
    int16_t* audio_buffer_10s = (int16_t*)malloc(TOTAL_SAMPLES * sizeof(int16_t));
    uint32_t* frame_buffer = (uint32_t*)malloc(FRAME_SIZE_BYTES);

    if (!audio_buffer_10s || !frame_buffer) {
        fprintf(stderr, "FATAL: Could not allocate main buffers. Exiting.\n");
        if (audio_buffer_10s) free(audio_buffer_10s);
        if (frame_buffer) free(frame_buffer);
        return;
    }

    while (1) {
        printf("----------------------------------------\n");
        printf("Connecting to server at %s:%d...\n", SERVER_IP, SERVER_PORT);
        int sock = socket_connect(SERVER_IP, SERVER_PORT);

        if (sock < 0) {
            printf("Connection failed. Retrying in 5 seconds...\n");
            sleep(5);
            continue;
        }

        printf("Connection successful. Collecting data for a new 10-second chunk...\n");
        for (int i = 0; i < NUM_FRAMES_TO_ACCUMULATE; ++i) {
            int16_t* buffer_offset = audio_buffer_10s + (i * SAMPLES_PER_FRAME);
            int status = receive_one_frame_and_extract(sock, buffer_offset, frame_buffer);
            if (status < 0) {
                break; // Socket error, break inner loop to reconnect
            }
        }

        printf("Buffer full. Processing data...\n");

        AudioFeatures* features = calculate_features(audio_buffer_10s, TOTAL_SAMPLES);
        if (features) {
            run_inference(features);
            cleanup_features(features);
        }

        socket_disconnect(sock);
        printf("Cycle complete. Waiting 5 seconds before next cycle...\n");
        sleep(5);
    }

    free(audio_buffer_10s);
    free(frame_buffer);
}

int main() {
    process_stream();
    return 0;
}
