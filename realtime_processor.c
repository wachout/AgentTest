#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include "socket_client.h"
#include "mfcc.h"
#include "feature_postprocessing.h"
#include "types.h"

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
// (AudioFeatures is now defined in mfcc.h)

// --- Function Declarations ---
int receive_one_frame_and_extract(int sockfd, int16_t* out_buffer, uint32_t* raw_frame_buffer);
AudioFeatures* calculate_features(const int16_t* audio_buffer, int num_samples);
int run_inference(const ProcessedFeatures* features); // Now takes ProcessedFeatures
void cleanup_features(AudioFeatures* features); // Renamed to avoid confusion

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

int run_inference(const ProcessedFeatures* features) {
    if (!features) return -1;
    printf("--- Inference Simulation ---\n");
    printf("Successfully received processed features for model.\n");
    printf("Mel Spec (size %d, first 5): [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
           TARGET_SIZE_MELSCRIPT, features->mel_spectrogram[0], features->mel_spectrogram[1], features->mel_spectrogram[2], features->mel_spectrogram[3], features->mel_spectrogram[4]);
    printf("STFT Spec (size %d, first 5): [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
           TARGET_SIZE_STFT, features->power_spectrogram[0], features->power_spectrogram[1], features->power_spectrogram[2], features->power_spectrogram[3], features->power_spectrogram[4]);
    printf("MFCC (size %d, first 5):      [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
           TARGET_SIZE_MFCC, features->mfcc[0], features->mfcc[1], features->mfcc[2], features->mfcc[3], features->mfcc[4]);
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

        printf("Connection successful. Collecting 10s of audio...\n");
        for (int i = 0; i < NUM_FRAMES_TO_ACCUMULATE; ++i) {
            int16_t* buffer_offset = audio_buffer_10s + (i * SAMPLES_PER_FRAME);
            int status = receive_one_frame_and_extract(sock, buffer_offset, frame_buffer);
            if (status < 0) {
                socket_disconnect(sock);
                goto reconnect; // Use goto to jump to the reconnect logic
            }
        }

        printf("Buffer full. Calculating raw features...\n");
        AudioFeatures* raw_features = calculate_features(audio_buffer_10s, TOTAL_SAMPLES);

        if (raw_features) {
            printf("\n--- Raw Feature Sizes ---\n");
            long spec_elements = (long)raw_features->num_frames * raw_features->num_spec_bins;
            long mel_elements = (long)raw_features->num_frames * raw_features->num_mels;
            long mfcc_elements = (long)raw_features->num_frames * raw_features->num_mfcc;
            printf("Power Spectrogram: %d frames x %d bins = %ld elements (%ld bytes)\n",
                   raw_features->num_frames, raw_features->num_spec_bins, spec_elements, spec_elements * sizeof(double));
            printf("Mel Spectrogram:   %d frames x %d bins = %ld elements (%ld bytes)\n",
                   raw_features->num_frames, raw_features->num_mels, mel_elements, mel_elements * sizeof(double));
            printf("MFCC:              %d frames x %d bins = %ld elements (%ld bytes)\n",
                   raw_features->num_frames, raw_features->num_mfcc, mfcc_elements, mfcc_elements * sizeof(double));

            printf("\nPost-processing features...\n");
            ProcessedFeatures* processed_features = process_features(raw_features);

            if (processed_features) {
                run_inference(processed_features);
                cleanup_processed_features(processed_features);
            }
            cleanup_features(raw_features);
        }

        socket_disconnect(sock);
        printf("Cycle complete.\n");

    reconnect:
        printf("Waiting 5 seconds before next cycle...\n");
        sleep(5);
    }

    free(audio_buffer_10s);
    free(frame_buffer);
}

int main() {
    process_stream();
    return 0;
}
