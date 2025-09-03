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

    // 2. Set up MFCC parameters
    mfcc_params params;
    params.sample_rate = wav_data.sample_rate;
    params.n_fft = 2048;
    params.hop_length = 512;
    params.n_mels = 128;
    params.n_mfcc = 13;
    params.fmin = 0.0;
    params.fmax = wav_data.sample_rate / 2.0;

    printf("\nComputing MFCCs with parameters:\n");
    printf("  - n_fft: %d\n", params.n_fft);
    printf("  - hop_length: %d\n", params.hop_length);
    printf("  - n_mels: %d\n", params.n_mels);
    printf("  - n_mfcc: %d\n", params.n_mfcc);

    // 3. Compute MFCCs
    int num_frames = 1 + (wav_data.num_samples - params.n_fft) / params.hop_length;
    if (num_frames <= 0) {
        fprintf(stderr, "Not enough audio data to compute a single frame.\n");
        free_wav_data(&wav_data);
        return 1;
    }

    double* mfcc_output = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));
    if (!mfcc_output) {
        fprintf(stderr, "Failed to allocate memory for MFCC output.\n");
        free_wav_data(&wav_data);
        return 1;
    }

    int frames_computed = compute_mfcc(&params, wav_data.audio_data, wav_data.num_samples, mfcc_output);
    if (frames_computed <= 0) {
        fprintf(stderr, "MFCC computation failed or produced no frames.\n");
        free(mfcc_output);
        free_wav_data(&wav_data);
        return 1;
    }

    printf("\nSuccessfully computed %d frames of MFCCs.\n", frames_computed);

    // 4. Print the results (first 5 frames)
    printf("\n--- MFCC Output (first 5 frames) ---\n");
    for (int i = 0; i < frames_computed && i < 5; ++i) {
        printf("Frame %d: [", i);
        for (int j = 0; j < params.n_mfcc; ++j) {
            printf("%8.4f", mfcc_output[i * params.n_mfcc + j]);
            if (j < params.n_mfcc - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    // 5. Clean up
    free(mfcc_output);
    free_wav_data(&wav_data);

    return 0;
}
