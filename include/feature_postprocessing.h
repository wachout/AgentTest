#ifndef FEATURE_POSTPROCESSING_H
#define FEATURE_POSTPROCESSING_H

#include "types.h" // Include the new central types header

// --- Target sizes for resampling ---
#define TARGET_SIZE_MELSCRIPT (38528)
#define TARGET_SIZE_STFT (77357)
#define TARGET_SIZE_MFCC (1880)

/**
 * @brief Processes raw audio features (resampling and normalization).
 *
 * @param raw_features A pointer to the AudioFeatures struct containing the raw data.
 * @return A pointer to a new ProcessedFeatures struct with the final data,
 *         or NULL on failure.
 */
ProcessedFeatures* process_features(const AudioFeatures* raw_features);

/**
 * @brief Frees all memory associated with a ProcessedFeatures struct.
 *
 * @param processed_features A pointer to the ProcessedFeatures struct to be freed.
 */
void cleanup_processed_features(ProcessedFeatures* processed_features);


#endif // FEATURE_POSTPROCESSING_H
