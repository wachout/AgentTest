#include "feature_postprocessing.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// --- Core Logic Implementation ---

/**
 * @brief Flattens, resamples, and normalizes a 2D feature matrix into a 1D vector.
 */
static void resample_and_normalize(const double* input_2d, int num_frames, int num_bins,
                                   double* output_1d, int target_size) {
    long total_elements = (long)num_frames * num_bins;

    // Create a temporary buffer to hold the resampled data before normalization
    double* resampled_data = (double*)calloc(target_size, sizeof(double));
    if (!resampled_data) {
        fprintf(stderr, "Failed to allocate temp buffer for resampling.\n");
        // Output will be all zeros
        return;
    }

    long num_elements_to_process = 0;

    // --- Step 1: Uniform Sampling (Resampling) & Padding ---
    if (total_elements > target_size) {
        // Downsample
        num_elements_to_process = target_size;
        double step = (double)total_elements / target_size;
        for (int i = 0; i < target_size; ++i) {
            long index = (long)floor(i * step);
            if (index < total_elements) {
                resampled_data[i] = input_2d[index];
            }
        }
    } else {
        // Copy all available data and leave the rest as zero-padding
        num_elements_to_process = total_elements;
        memcpy(resampled_data, input_2d, total_elements * sizeof(double));
    }

    if (num_elements_to_process == 0) {
        free(resampled_data);
        return; // Nothing to normalize
    }

    // --- Step 2: Normalization (Mean and Variance) ---
    // Calculate mean
    double sum = 0.0;
    for (long i = 0; i < num_elements_to_process; ++i) {
        sum += resampled_data[i];
    }
    double mean = sum / num_elements_to_process;

    // Calculate variance and standard deviation
    double sum_sq_diff = 0.0;
    for (long i = 0; i < num_elements_to_process; ++i) {
        sum_sq_diff += (resampled_data[i] - mean) * (resampled_data[i] - mean);
    }
    double std_dev = sqrt(sum_sq_diff / num_elements_to_process);

    // Apply normalization to the final output buffer
    // If std_dev is very small, normalized values will be 0
    if (std_dev < 1e-9) {
        // All elements are the same, so (x - mean) is 0.
        // The output buffer is already zeroed by calloc, so we are done.
    } else {
        for (long i = 0; i < num_elements_to_process; ++i) {
            output_1d[i] = (resampled_data[i] - mean) / std_dev;
        }
    }

    free(resampled_data);
}


// --- Public API ---

ProcessedFeatures* process_features(const AudioFeatures* raw_features) {
    if (!raw_features) {
        return NULL;
    }

    ProcessedFeatures* processed = (ProcessedFeatures*)malloc(sizeof(ProcessedFeatures));
    if (!processed) {
        fprintf(stderr, "Failed to allocate memory for ProcessedFeatures struct.\n");
        return NULL;
    }

    processed->mel_spectrogram = (double*)calloc(TARGET_SIZE_MELSCRIPT, sizeof(double));
    processed->power_spectrogram = (double*)calloc(TARGET_SIZE_STFT, sizeof(double));
    processed->mfcc = (double*)calloc(TARGET_SIZE_MFCC, sizeof(double));

    if (!processed->mel_spectrogram || !processed->power_spectrogram || !processed->mfcc) {
        fprintf(stderr, "Failed to allocate memory for processed feature buffers.\n");
        cleanup_processed_features(processed);
        return NULL;
    }

    printf("Post-processing Mel Spectrogram...\n");
    resample_and_normalize(raw_features->mel_spectrogram, raw_features->num_frames, raw_features->num_mels,
                           processed->mel_spectrogram, TARGET_SIZE_MELSCRIPT);

    printf("Post-processing Power Spectrogram...\n");
    resample_and_normalize(raw_features->power_spectrogram, raw_features->num_frames, raw_features->num_spec_bins,
                           processed->power_spectrogram, TARGET_SIZE_STFT);

    printf("Post-processing MFCC...\n");
    resample_and_normalize(raw_features->mfcc, raw_features->num_frames, raw_features->num_mfcc,
                           processed->mfcc, TARGET_SIZE_MFCC);

    return processed;
}

void cleanup_processed_features(ProcessedFeatures* processed_features) {
    if (processed_features) {
        if (processed_features->mel_spectrogram) free(processed_features->mel_spectrogram);
        if (processed_features->power_spectrogram) free(processed_features->power_spectrogram);
        if (processed_features->mfcc) free(processed_features->mfcc);
        free(processed_features);
    }
}
