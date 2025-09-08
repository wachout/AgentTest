#ifndef MFCC_H
#define MFCC_H

#include <stdlib.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold MFCC parameters
typedef struct {
    int n_mfcc;
    int n_mels;
    int n_fft;
    int hop_length;
    double fmin;
    double fmax;
    int sample_rate;
} mfcc_params;

/**
 * @brief Computes the Mel Frequency Cepstral Coefficients (MFCCs) for an audio signal.
 *
 * @param params Parameters for MFCC computation.
 * @param audio_buffer The input audio signal.
 * @param buffer_len The length of the audio buffer.
 * @param[out] mfcc_output The buffer to store the computed MFCCs.
 *                         The size should be (num_frames * n_mfcc).
 * @return The number of frames processed.
 */
int compute_mfcc(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* mfcc_output);

/**
 * @brief Computes the Mel Spectrogram for an audio signal.
 *
 * @param params Parameters for the computation (uses all except n_mfcc).
 * @param audio_buffer The input audio signal.
 * @param buffer_len The length of the audio buffer.
 * @param[out] melspec_output The buffer to store the computed spectrogram.
 *                            The size should be (num_frames * n_mels).
 * @return The number of frames processed, or -1 on error.
 */
int compute_melspectrogram(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* melspec_output);

/**
 * @brief Computes the Power Spectrogram (STFT squared magnitude) for an audio signal.
 *
 * @param params Parameters for the computation.
 * @param audio_buffer The input audio signal.
 * @param buffer_len The length of the audio buffer.
 * @param[out] spec_output The buffer to store the computed spectrogram.
 *                         The size should be (num_frames * (fft_size/2 + 1)).
 * @return The number of frames processed, or -1 on error.
 */
int compute_power_spectrogram(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* spec_output);

/**
 * @brief Helper function to find the next power of 2 for a given number.
 *
 * @param n The input number.
 * @return The smallest power of 2 that is greater than or equal to n.
 */
unsigned int next_power_of_2(unsigned int n);

/**
 * @brief Computes one or more features in a single, efficient pass.
 *
 * @param params Parameters for the computation.
 * @param audio_buffer The input audio signal.
 * @param buffer_len The length of the audio buffer.
 * @param[out] power_spec_output (Optional, can be NULL) Buffer for the power spectrogram.
 * @param[out] melspec_output (Optional, can be NULL) Buffer for the Mel spectrogram.
 * @param[out] mfcc_output (Optional, can be NULL) Buffer for the MFCC features.
 * @return The number of frames processed, or -1 on error.
 */
int compute_features(mfcc_params* params, const double* audio_buffer, size_t buffer_len,
                     double* power_spec_output, double* melspec_output, double* mfcc_output);


/**
 * @brief Generates a Mel filterbank. The caller is responsible for allocating and freeing the filterbank memory.
 *
 * @param filterbank Pointer to an array of pointers (2D array) to store the filterbank.
 *                   The shape should be (n_mels, n_fft / 2 + 1).
 * @param n_mels Number of Mel filters.
 * @param n_fft FFT size.
 * @param sample_rate Sample rate of the audio.
 * @param f_min Minimum frequency.
 * @param f_max Maximum frequency.
 */
void generate_mel_filterbank(double** filterbank, int n_mels, int n_fft, int sample_rate, double f_min, double f_max);

/**
 * @brief Computes the Discrete Cosine Transform (Type II, ortho-normalized).
 *
 * @param input Input vector of size num_mels.
 * @param output Output vector of size num_mfcc.
 * @param num_mels The number of Mel filters (input size).
 * @param num_mfcc The number of MFCC coefficients to compute (output size).
 */
void dct2(const double* input, double* output, int num_mels, int num_mfcc);

/**
 * @brief Converts a power spectrogram to the decibel scale, in-place.
 *        This mimics librosa's power_to_db(S, ref=np.max).
 *
 * @param spec The power spectrogram to convert.
 * @param len The length of the spectrogram array.
 */
void mfcc_power_to_db(double* spec, int len);


#ifdef __cplusplus
}
#endif

#endif // MFCC_H
