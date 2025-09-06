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

    WavData wav_data;
    if (load_wav_file(wav_filename, &wav_data) != 0) {
        fprintf(stderr, "Failed to load WAV file.\n");
        return 1;
    }

    printf("WAV loaded successfully:\n");
    printf("  - Sample Rate: %u\n", wav_data.sample_rate);
    printf("  - Samples: %u\n", wav_data.num_samples);

    if (wav_data.num_samples == 0) {
        printf("WAV file contains no audio data.\n");
        free_wav_data(&wav_data);
        return 0;
    }

    mfcc_params params;
    params.sample_rate = wav_data.sample_rate;
    params.n_fft = 512;
    params.hop_length = 160;
    params.n_mels = 128;
    params.n_mfcc = 20;
    params.fmin = 0.0;
    params.fmax = wav_data.sample_rate / 2.0;

    int num_frames = 1 + (wav_data.num_samples - params.n_fft) / params.hop_length;
    if (num_frames <= 0) {
        fprintf(stderr, "Not enough audio data to compute a single frame.\n");
        free_wav_data(&wav_data);
        return 1;
    }

    // --- Efficiently compute both MFCC and Melspectrogram in one pass ---
    printf("\nComputing all features in a single pass...\n");

    double* mfcc_output = (double*)malloc(num_frames * params.n_mfcc * sizeof(double));
    double* melspec_output = (double*)malloc(num_frames * params.n_mels * sizeof(double));

    if (!mfcc_output || !melspec_output) {
        fprintf(stderr, "Failed to allocate memory for feature output.\n");
        free_wav_data(&wav_data);
        if (mfcc_output) free(mfcc_output);
        if (melspec_output) free(melspec_output);
        return 1;
    }

    // Call the new high-level function. We pass NULL for the power spectrogram
    // because we don't need it for this example.
    int frames_computed = compute_features(&params, wav_data.audio_data, wav_data.num_samples,
                                           NULL, melspec_output, mfcc_output);

    if (frames_computed > 0) {
        printf("Successfully computed %d frames.\n\n", frames_computed);

        // Print MFCC summary
        printf("--- MFCC Output (first 5 frames) ---\n");
        for (int i = 0; i < frames_computed && i < 5; ++i) {
            printf("Frame %d: [", i);
            for (int j = 0; j < params.n_mfcc; ++j) {
                printf("%8.2f", mfcc_output[i * params.n_mfcc + j]);
                if (j < params.n_mfcc - 1) printf(", ");
            }
            printf("]\n");
        }

        // Print Mel Spectrogram summary
        printf("\n--- Mel Spectrogram Output (first frame, first 5 values) ---\n");
        printf("Frame 0: [");
        for (int i = 0; i < 5; ++i) {
            printf("%.4f", melspec_output[i]);
            if (i < 4) printf(", ");
        }
        printf("...]\n");

    } else {
        fprintf(stderr, "Feature computation failed.\n");
    }

    // Final Cleanup
    free(mfcc_output);
    free(melspec_output);
    free_wav_data(&wav_data);

    return 0;
}
