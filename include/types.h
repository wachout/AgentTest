#ifndef AUDIO_TYPES_H
#define AUDIO_TYPES_H

#include <stdlib.h>

// --- Data Structures ---

/**
 * @brief Holds the raw, unprocessed feature matrices.
 */
typedef struct {
    int num_frames;
    int num_spec_bins;
    int num_mels;
    int num_mfcc;
    double* power_spectrogram;
    double* mel_spectrogram;
    double* mfcc;
} AudioFeatures;

/**
 * @brief Holds the final, post-processed feature vectors.
 *
 * Each vector is flattened, uniformly sampled, and normalized.
 */
typedef struct {
    double* mel_spectrogram;
    double* power_spectrogram;
    double* mfcc;
} ProcessedFeatures;

#endif // AUDIO_TYPES_H
