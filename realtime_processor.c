#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include "socket_client.h"
#include "mfcc.h"

// Configuration
#define SERVER_IP "192.168.35.6"
#define SERVER_PORT 8888
#define SAMPLES_PER_FRAME 16384
#define NUM_CHANNELS 64
#define FRAME_SIZE_BYTES (SAMPLES_PER_FRAME * NUM_CHANNELS * sizeof(uint32_t))
#define NUM_FRAMES_TO_ACCUMULATE 59
#define TOTAL_SAMPLES (NUM_FRAMES_TO_ACCUMULATE * SAMPLES_PER_FRAME)
#define SAMPLE_RATE 96000

// Main processing function for one 10-second batch
void process_batch(int sockfd) {
    // 1. Allocate buffers
    uint32_t* raw_frame_buffer = (uint32_t*)malloc(FRAME_SIZE_BYTES);
    int16_t* accumulated_int16_buffer = (int16_t*)malloc(TOTAL_SAMPLES * sizeof(int16_t));
    double* final_audio_buffer = (double*)malloc(TOTAL_SAMPLES * sizeof(double));
    if (!raw_frame_buffer || !accumulated_int16_buffer || !final_audio_buffer) {
        fprintf(stderr, "Failed to allocate memory for processing buffers.\n");
        if (raw_frame_buffer) free(raw_frame_buffer);
        if (accumulated_int16_buffer) free(accumulated_int16_buffer);
        if (final_audio_buffer) free(final_audio_buffer);
        return;
    }

    // 2. Read frames and prepare data
    printf("Accumulating %d frames...\n", NUM_FRAMES_TO_ACCUMULATE);
    for (int i = 0; i < NUM_FRAMES_TO_ACCUMULATE; ++i) {
        if (socket_read_fully(sockfd, raw_frame_buffer, FRAME_SIZE_BYTES) != 0) {
            fprintf(stderr, "Failed to read full frame %d. Disconnecting.\n", i);
            goto cleanup;
        }
        for (int j = 0; j < SAMPLES_PER_FRAME; ++j) {
            uint32_t sample_uint32 = raw_frame_buffer[j * NUM_CHANNELS];
            int32_t sample_int32 = (int32_t)(sample_uint32 - 2147483648U);
            accumulated_int16_buffer[i * SAMPLES_PER_FRAME + j] = (int16_t)(sample_int32 >> 16);
        }
    }
    printf("Accumulation complete. Converting to double...\n");
    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        final_audio_buffer[i] = (double)accumulated_int16_buffer[i] / 32768.0;
    }

    // 3. Set up parameters
    mfcc_params params;
    params.sample_rate = SAMPLE_RATE;
    params.n_fft = 512;
    params.hop_length = 160;
    params.n_mels = 128;
    params.n_mfcc = 20;
    params.fmin = 0.0;
    params.fmax = SAMPLE_RATE / 2.0;

    int num_frames = 1 + (TOTAL_SAMPLES - params.n_fft) / params.hop_length;
    unsigned int spec_len = next_power_of_2(params.n_fft) / 2 + 1;

    // 4. Allocate output buffers
    double* spec_output = (double*)malloc(num_frames * spec_len * sizeof(double));
    double* melspec_output = (double*)malloc(num_frames * params.n_mels * sizeof(double));
    double* mfcc_output = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));

    // 5. Extract all features in one go
    printf("\n--- Extracting All Features (Optimized) ---\n");
    if (compute_features(&params, final_audio_buffer, TOTAL_SAMPLES, spec_output, melspec_output, mfcc_output) > 0) {
        printf("Power Spectrogram (Frame 0, first 5): [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
               spec_output[0], spec_output[1], spec_output[2], spec_output[3], spec_output[4]);
        printf("Mel Spectrogram (Frame 0, first 5):   [%.4f, %.4f, %.4f, %.4f, %.4f...]\n",
               melspec_output[0], melspec_output[1], melspec_output[2], melspec_output[3], melspec_output[4]);
        printf("MFCC (Frame 0, first 5):            [%.2f, %.2f, %.2f, %.2f, %.2f...]\n",
               mfcc_output[0], mfcc_output[1], mfcc_output[2], mfcc_output[3], mfcc_output[4]);
    } else {
        fprintf(stderr, "Feature computation failed.\n");
    }

    // 6. Cleanup
    free(spec_output);
    free(melspec_output);
    free(mfcc_output);

cleanup:
    free(raw_frame_buffer);
    free(accumulated_int16_buffer);
    free(final_audio_buffer);
}

int main() {
    printf("Starting real-time processor client...\n");
    while (1) {
        printf("----------------------------------------\n");
        printf("Connecting to server at %s:%d...\n", SERVER_IP, SERVER_PORT);
        int sockfd = socket_connect(SERVER_IP, SERVER_PORT);
        if (sockfd >= 0) {
            printf("Connection successful. Starting processing batch.\n");
            process_batch(sockfd);
            socket_disconnect(sockfd);
            printf("Batch finished. Disconnected.\n");
        } else {
            printf("Connection failed.\n");
        }
        printf("Waiting 5 seconds before next cycle...\n");
        sleep(5);
    }
    return 0;
}
