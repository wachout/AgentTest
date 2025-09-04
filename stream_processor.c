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
    uint32_t* frame_buffer = (uint32_t*)malloc(SAMPLES_PER_FRAME * NUM_CHANNELS * sizeof(uint32_t));
    if (!accumulated_buffer || !frame_buffer) {
        fprintf(stderr, "Failed to allocate memory for buffers.\n");
        if (accumulated_buffer) free(accumulated_buffer);
        if (frame_buffer) free(frame_buffer);
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
    free(frame_buffer);

    printf("Data is ready for feature extraction.\n");

    // Set up parameters
    mfcc_params params;
    params.sample_rate = SAMPLE_RATE;
    params.n_fft = 2048;
    params.hop_length = 512;
    params.n_mels = 128;
    params.n_mfcc = 13;
    params.fmin = 0.0;
    params.fmax = SAMPLE_RATE / 2.0;

    int num_frames = 1 + (TOTAL_SAMPLES_SINGLE_CHANNEL - params.n_fft) / params.hop_length;
    if (num_frames <= 0) {
        fprintf(stderr, "Not enough audio data to compute a single frame.\n");
        free(accumulated_buffer);
        return 1;
    }

    // --- MFCC Extraction ---
    printf("\nComputing MFCCs...\n");
    double* mfcc_output = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));
    if (!mfcc_output) {
        fprintf(stderr, "Failed to allocate memory for MFCC output.\n");
        free(accumulated_buffer);
        return 1;
    }
    int mfcc_frames_computed = compute_mfcc(&params, accumulated_buffer, TOTAL_SAMPLES_SINGLE_CHANNEL, mfcc_output);
    if (mfcc_frames_computed > 0) {
        printf("Successfully computed %d frames of MFCCs.\n", mfcc_frames_computed);
        printf("--- MFCC Output (first 5 frames) ---\n");
        for (int i = 0; i < mfcc_frames_computed && i < 5; ++i) {
            printf("Frame %d: [", i);
            for (int j = 0; j < params.n_mfcc; ++j) {
                printf("%8.4f", mfcc_output[i * params.n_mfcc + j]);
                if (j < params.n_mfcc - 1) printf(", ");
            }
            printf("]\n");
        }
    } else {
        fprintf(stderr, "MFCC computation failed.\n");
    }
    free(mfcc_output);

    // --- Mel Spectrogram Extraction ---
    printf("\nComputing Mel Spectrogram...\n");
    double* melspec_output = (double*)malloc(num_frames * params.n_mels * sizeof(double));
    if (!melspec_output) {
        fprintf(stderr, "Failed to allocate memory for Mel Spectrogram output.\n");
        free(accumulated_buffer);
        return 1;
    }
    int melspec_frames_computed = compute_melspectrogram(&params, accumulated_buffer, TOTAL_SAMPLES_SINGLE_CHANNEL, melspec_output);
    if (melspec_frames_computed > 0) {
        printf("Successfully computed %d frames of Mel Spectrogram.\n", melspec_frames_computed);
        printf("--- Mel Spectrogram Output (first frame, first 5 values) ---\n");
        printf("Frame 0: [");
        for (int i = 0; i < 5; ++i) {
            printf("%.8f", melspec_output[i]);
            if (i < 4) printf(", ");
        }
        printf("...]\n");
    } else {
        fprintf(stderr, "Mel Spectrogram computation failed.\n");
    }
    free(melspec_output);

    // Final cleanup
    free(accumulated_buffer);
    return 0;
}
