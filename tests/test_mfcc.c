#include "mfcc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define AUDIO_LEN 96000
#define NUM_FRAMES 184
#define N_MFCC 13
// Set a reasonable tolerance for the full pipeline comparison
#define TOLERANCE 1e-2

// Helper function to read a binary file into a double array
long read_bin_file(const char* filename, double* buffer, long max_len) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return -1;
    }
    long items_read = fread(buffer, sizeof(double), max_len, f);
    fclose(f);
    return items_read;
}

int main() {
    printf("\n--- Running Full End-to-End MFCC Test ---\n");

    // 1. Set up parameters
    mfcc_params params;
    params.sample_rate = 96000;
    params.n_fft = 2048;
    params.hop_length = 512;
    params.n_mels = 128;
    params.n_mfcc = N_MFCC;
    params.fmin = 0.0;
    params.fmax = 48000.0;

    // 2. Allocate memory and read test data
    double* audio_buffer = (double*)malloc(sizeof(double) * AUDIO_LEN);
    double* librosa_mfccs = (double*)malloc(sizeof(double) * NUM_FRAMES * N_MFCC);
    double* c_mfccs = (double*)malloc(sizeof(double) * NUM_FRAMES * N_MFCC);

    if (!audio_buffer || !librosa_mfccs || !c_mfccs) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    if (read_bin_file("test_signal.bin", audio_buffer, AUDIO_LEN) != AUDIO_LEN) {
        fprintf(stderr, "Failed to read test_signal.bin\n");
        return 1;
    }

    if (read_bin_file("librosa_mfccs.bin", librosa_mfccs, NUM_FRAMES * N_MFCC) != NUM_FRAMES * N_MFCC) {
        fprintf(stderr, "Failed to read librosa_mfccs.bin\n");
        return 1;
    }

    // 3. Run our full MFCC implementation
    int num_frames_processed = compute_mfcc(&params, audio_buffer, AUDIO_LEN, c_mfccs);

    if (num_frames_processed != NUM_FRAMES) {
        fprintf(stderr, "Test failed: Number of frames do not match. Expected %d, got %d\n", NUM_FRAMES, num_frames_processed);
        return 1;
    }

    // 4. Compare results
    int errors = 0;
    for (int i = 0; i < NUM_FRAMES * N_MFCC; ++i) {
        double diff = fabs(c_mfccs[i] - librosa_mfccs[i]);
        if (diff > TOLERANCE) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at index %d: C_MFCC=%.8f, Librosa_MFCC=%.8f, Diff=%.8f\n",
                        i, c_mfccs[i], librosa_mfccs[i], diff);
            }
            errors++;
        }
    }

    // 5. Report result
    if (errors == 0) {
        printf("TEST PASSED!\n");
    } else {
        fprintf(stderr, "TEST FAILED: %d errors found with tolerance %.6f.\n", errors, TOLERANCE);
    }

    free(audio_buffer);
    free(librosa_mfccs);
    free(c_mfccs);

    return (errors == 0) ? 0 : 1;
}
