#ifndef WAV_UTILS_H
#define WAV_UTILS_H

#include <stdint.h>

typedef struct {
    double* audio_data;    // Pointer to the audio data (first channel only)
    uint32_t num_samples;  // Total number of samples in the buffer
    uint32_t sample_rate;  // Sample rate of the audio
    uint16_t num_channels; // Number of channels in the original file
} WavData;

/**
 * @brief Loads a WAV file and returns its audio data.
 *
 * This function reads a WAV file, extracts the first channel, and converts
 * the samples to normalized `double` format [-1.0, 1.0].
 * The caller is responsible for freeing the `audio_data` buffer in the
 * returned struct.
 *
 * @param filename The path to the WAV file.
 * @param[out] wav_data A pointer to a WavData struct to be populated.
 * @return 0 on success, -1 on failure.
 */
int load_wav_file(const char* filename, WavData* wav_data);

/**
 * @brief Frees the memory allocated for WavData.
 *
 * @param wav_data A pointer to the WavData struct to be freed.
 */
void free_wav_data(WavData* wav_data);


#endif // WAV_UTILS_H
