#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mfcc.h"

// Stream parameters
#define SAMPLES_PER_FRAME 16384
#define NUM_CHANNELS 64
#define NUM_FRAMES_TO_PROCESS 2
#define SAMPLE_RATE 96000 // Derived from 16384 samples / 0.170s

// Total number of samples for a single channel after accumulation
#define TOTAL_SAMPLES_SINGLE_CHANNEL (NUM_FRAMES_TO_PROCESS * SAMPLES_PER_FRAME)

// Normalization constant for uint32 -> double
#define UINT32_TO_DOUBLE_NORM 2147483648.0

int main() {
    const char* stream_filename = "stream_data.bin";
    FILE* f = fopen(stream_filename, "rb");
    if (!f) {
        perror("Failed to open stream_data.bin");
        return 1;
    }

    printf("Processing data stream from: %s\n", stream_filename);

    double* accumulated_buffer = (double*)malloc(TOTAL_SAMPLES_SINGLE_CHANNEL * sizeof(double));
    if (!accumulated_buffer) {
        fprintf(stderr, "Failed to allocate memory for accumulated data.\n");
        fclose(f);
        return 1;
    }

    uint32_t* frame_buffer = (uint32_t*)malloc(SAMPLES_PER_FRAME * NUM_CHANNELS * sizeof(uint32_t));
    if (!frame_buffer) {
        fprintf(stderr, "Failed to allocate memory for a single frame.\n");
        free(accumulated_buffer);
        fclose(f);
        return 1;
    }

    printf("Reading and processing %d frames...\n", NUM_FRAMES_TO_PROCESS);

    for (int i = 0; i < NUM_FRAMES_TO_PROCESS; ++i) {
        size_t items_read = fread(frame_buffer, sizeof(uint32_t), SAMPLES_PER_FRAME * NUM_CHANNELS, f);
        if (items_read != SAMPLES_PER_FRAME * NUM_CHANNELS) {
            fprintf(stderr, "Error reading frame %d from file.\n", i);
            free(accumulated_buffer);
            free(frame_buffer);
            fclose(f);
            return 1;
        }

        for (int j = 0; j < SAMPLES_PER_FRAME; ++j) {
            uint32_t sample_uint32 = frame_buffer[j * NUM_CHANNELS];
            double sample_double = ((double)sample_uint32 - UINT32_TO_DOUBLE_NORM) / UINT32_TO_DOUBLE_NORM;
            accumulated_buffer[i * SAMPLES_PER_FRAME + j] = sample_double;
        }
    }
    fclose(f);
    free(frame_buffer); // No longer needed

    printf("Data is ready for MFCC processing.\n");

    // 1. Set up MFCC parameters
    mfcc_params params;
    params.sample_rate = SAMPLE_RATE;
    params.n_fft = 2048;
    params.hop_length = 512;
    params.n_mels = 128;
    params.n_mfcc = 13;
    params.fmin = 0.0;
    params.fmax = SAMPLE_RATE / 2.0;

    // 2. Allocate MFCC output buffer
    int num_mfcc_frames = 1 + (TOTAL_SAMPLES_SINGLE_CHANNEL - params.n_fft) / params.hop_length;
    if (num_mfcc_frames <= 0) {
        fprintf(stderr, "Not enough audio data to compute a single MFCC frame.\n");
        free(accumulated_buffer);
        return 1;
    }
    double* mfcc_output = (double*)malloc(num_mfcc_frames * params.n_mfcc * sizeof(double));
    if (!mfcc_output) {
        fprintf(stderr, "Failed to allocate memory for MFCC output.\n");
        free(accumulated_buffer);
        return 1;
    }

    // 3. Call compute_mfcc
    printf("Computing MFCCs...\n");
    int frames_computed = compute_mfcc(&params, accumulated_buffer, TOTAL_SAMPLES_SINGLE_CHANNEL, mfcc_output);
    if (frames_computed <= 0) {
        fprintf(stderr, "MFCC computation failed.\n");
        free(accumulated_buffer);
        free(mfcc_output);
        return 1;
    }

    printf("\nSuccessfully computed %d frames of MFCCs.\n", frames_computed);

    // 4. Print the results (first 5 frames)
    printf("\n--- MFCC Output (first 5 frames) ---\n");
    for (int i = 0; i < frames_computed && i < 5; ++i) {
        printf("Frame %d: [", i);
        for (int j = 0; j < params.n_mfcc; ++j) {
            printf("%8.4f", mfcc_output[i * params.n_mfcc + j]);
            if (j < params.n_mfcc - 1) printf(", ");
        }
        printf("]\n");
    }

    // 5. Clean up
    free(accumulated_buffer);
    free(mfcc_output);

    return 0;
}
