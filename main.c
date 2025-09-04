#include <stdio.h>
#include <stdlib.h>
#include "wav_utils.h"
#include "mfcc.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path_to_wav_file>\n", argv[0]);
        return 1;
    }

    const char* wav_filename = argv[1];
    printf("Loading WAV file: %s\n", wav_filename);

    // 1. Load WAV file
    WavData wav_data;
    if (load_wav_file(wav_filename, &wav_data) != 0) {
        fprintf(stderr, "Failed to load WAV file.\n");
        return 1;
    }

    printf("WAV loaded successfully:\n");
    printf("  - Sample Rate: %u\n", wav_data.sample_rate);
    printf("  - Channels: %u\n", wav_data.num_channels);
    printf("  - Samples: %u\n", wav_data.num_samples);

    if (wav_data.num_samples == 0) {
        printf("WAV file contains no audio data.\n");
        free_wav_data(&wav_data);
        return 0;
    }

    // 2. Set up MFCC/Melspectrogram parameters
    mfcc_params params;
    params.sample_rate = wav_data.sample_rate;
    params.n_fft = 2048;
    params.hop_length = 512;
    params.n_mels = 128;
    params.n_mfcc = 13;
    params.fmin = 0.0;
    params.fmax = wav_data.sample_rate / 2.0;

    int num_frames = 1 + (wav_data.num_samples - params.n_fft) / params.hop_length;
    if (num_frames <= 0) {
        fprintf(stderr, "Not enough audio data to compute a single frame.\n");
        free_wav_data(&wav_data);
        return 1;
    }

    // --- MFCC Extraction ---
    printf("\nComputing MFCCs...\n");
    double* mfcc_output = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));
    if (!mfcc_output) {
        fprintf(stderr, "Failed to allocate memory for MFCC output.\n");
        free_wav_data(&wav_data);
        return 1;
    }
    int mfcc_frames_computed = compute_mfcc(&params, wav_data.audio_data, wav_data.num_samples, mfcc_output);
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
        free_wav_data(&wav_data);
        return 1;
    }
    int melspec_frames_computed = compute_melspectrogram(&params, wav_data.audio_data, wav_data.num_samples, melspec_output);
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

    // Final Cleanup
    free_wav_data(&wav_data);

    return 0;
}
