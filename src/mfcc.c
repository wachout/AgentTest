#include "mfcc.h"
#include "tinyfft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations for static helper functions
static void bit_reversal_permutation(cmplx* buffer, int k);
static unsigned int next_power_of_2(unsigned int n);

// Helper function to find the next power of 2
static unsigned int next_power_of_2(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Helper function to convert frequency from Hertz to Mel scale
static double hz_to_mel(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

// Helper function to convert frequency from Mel scale to Hertz
static double mel_to_hz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// Helper function to generate a Hann window
static void hann_window(double* window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (length - 1)));
    }
}

// Helper function to reverse bits for bit-reversal permutation
static unsigned int reverse_bits(unsigned int num, unsigned int k) {
    unsigned int reversed_num = 0;
    for (unsigned int i = 0; i < k; i++) {
        if ((num >> i) & 1) {
            reversed_num |= 1 << ((k - 1) - i);
        }
    }
    return reversed_num;
}

static void bit_reversal_permutation(cmplx* buffer, int k) {
    int n = 1 << k;
    for (int i = 0; i < n; i++) {
        unsigned int reversed_i = reverse_bits(i, k);
        if (i < reversed_i) {
            cmplx temp = buffer[i];
            buffer[i] = buffer[reversed_i];
            buffer[reversed_i] = temp;
        }
    }
}

void generate_mel_filterbank(double** filterbank, int n_mels, int n_fft, int sample_rate, double f_min, double f_max) {
    unsigned int fft_size = next_power_of_2(n_fft); // Use padded size for filterbank

    double min_mel = hz_to_mel(f_min);
    double max_mel = hz_to_mel(f_max);

    double* mel_points = (double*)malloc(sizeof(double) * (n_mels + 2));
    double* hz_points = (double*)malloc(sizeof(double) * (n_mels + 2));
    int* fft_bins = (int*)malloc(sizeof(int) * (n_mels + 2));

    double mel_step = (max_mel - min_mel) / (n_mels + 1);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = min_mel + i * mel_step;
        hz_points[i] = mel_to_hz(mel_points[i]);
        fft_bins[i] = floor((fft_size + 1) * hz_points[i] / sample_rate);
    }

    for (int i = 0; i < n_mels; ++i) {
        if (hz_points[i+2] == hz_points[i]) continue;
        double enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);

        int start_bin = fft_bins[i];
        int center_bin = fft_bins[i + 1];
        int end_bin = fft_bins[i + 2];

        for (int j = start_bin; j < center_bin; ++j) {
            if (j >= 0 && j < (fft_size / 2 + 1))
                filterbank[i][j] = (double)(j - start_bin) / (center_bin - start_bin) * enorm;
        }
        for (int j = center_bin; j < end_bin; ++j) {
            if (j >= 0 && j < (fft_size / 2 + 1))
                filterbank[i][j] = (double)(end_bin - j) / (end_bin - center_bin) * enorm;
        }
    }

    free(mel_points);
    free(hz_points);
    free(fft_bins);
}

void mfcc_power_to_db(double* spec, int len) {
    double amin = 1e-10;
    double top_db = 80.0;

    double max_val = 0.0;
    for (int i = 0; i < len; ++i) {
        if (spec[i] > max_val) max_val = spec[i];
    }

    double ref_value = max_val;

    for (int i = 0; i < len; ++i) {
        double s_val = (spec[i] > amin) ? spec[i] : amin;
        double r_val = (ref_value > amin) ? ref_value : amin;
        spec[i] = 10.0 * (log10(s_val) - log10(r_val));
    }

    double max_log_val = -1e100;
    for (int i = 0; i < len; ++i) {
        if (spec[i] > max_log_val) max_log_val = spec[i];
    }

    for (int i = 0; i < len; ++i) {
        if (spec[i] < max_log_val - top_db) {
            spec[i] = max_log_val - top_db;
        }
    }
}

void dct2(const double* input, double* output, int num_mels, int num_mfcc) {
    for (int k = 0; k < num_mfcc; ++k) {
        double sum = 0.0;
        for (int n = 0; n < num_mels; ++n) {
            sum += input[n] * cos(M_PI / num_mels * (n + 0.5) * k);
        }
        double scale = (k == 0) ? sqrt(1.0 / num_mels) : sqrt(2.0 / num_mels);
        output[k] = sum * scale;
    }
}

int compute_melspectrogram(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* melspec_output) {
    int n_fft = params->n_fft; // User-defined frame length
    int hop_length = params->hop_length;
    int n_mels = params->n_mels;

    // Calculate the FFT size as the next power of 2
    unsigned int fft_size = next_power_of_2(n_fft);

    if (buffer_len < (size_t)n_fft) return 0;
    int num_frames = 1 + (buffer_len - n_fft) / hop_length;

    double* window = (double*)malloc(sizeof(double) * n_fft);
    hann_window(window, n_fft);

    double** mel_filterbank = (double**)malloc(sizeof(double*) * n_mels);
    for (int i = 0; i < n_mels; i++) {
        mel_filterbank[i] = (double*)calloc(fft_size / 2 + 1, sizeof(double));
    }
    // Note: generate_mel_filterbank also needs to know the padded fft_size
    generate_mel_filterbank(mel_filterbank, n_mels, n_fft, params->sample_rate, params->fmin, params->fmax);

    int k = (int)log2(fft_size);
    cmplx* fft_twiddles = (cmplx*)malloc(sizeof(cmplx) * (1 << (k - 1)));
    tfft_init(k, fft_twiddles);

    double* frame = (double*)malloc(sizeof(double) * n_fft);
    cmplx* fft_buffer = (cmplx*)malloc(sizeof(cmplx) * fft_size);
    double* power_spec = (double*)malloc(sizeof(double) * (fft_size / 2 + 1));

    for (int i = 0; i < num_frames; ++i) {
        int start = i * hop_length;
        memcpy(frame, audio_buffer + start, sizeof(double) * n_fft);

        for (int j = 0; j < n_fft; ++j) frame[j] *= window[j];

        // Zero-pad the frame into the FFT buffer
        for (int j = 0; j < n_fft; ++j) fft_buffer[j] = frame[j] + 0.0 * I;
        for (unsigned int j = n_fft; j < fft_size; ++j) fft_buffer[j] = 0.0 + 0.0 * I;

        tfft_fft(k, fft_buffer, fft_twiddles);
        bit_reversal_permutation(fft_buffer, k);

        for (unsigned int j = 0; j < fft_size / 2 + 1; ++j) {
            power_spec[j] = (creal(fft_buffer[j]) * creal(fft_buffer[j]) + cimag(fft_buffer[j]) * cimag(fft_buffer[j]));
        }

        double* current_melspec_frame = melspec_output + (i * n_mels);
        for (int j = 0; j < n_mels; ++j) {
            current_melspec_frame[j] = 0.0;
            for (unsigned int bin = 0; bin < fft_size / 2 + 1; ++bin) {
                current_melspec_frame[j] += power_spec[bin] * mel_filterbank[j][bin];
            }
        }
    }

    free(window);
    for (int i = 0; i < n_mels; i++) free(mel_filterbank[i]);
    free(mel_filterbank);
    free(fft_twiddles);
    free(frame);
    free(fft_buffer);
    free(power_spec);

    return num_frames;
}

int compute_mfcc(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* mfcc_output) {
    int n_mels = params->n_mels;
    int n_mfcc = params->n_mfcc;

    if (buffer_len < (size_t)params->n_fft) return 0;
    int num_frames = 1 + (buffer_len - params->n_fft) / params->hop_length;

    double* melspec_buffer = (double*)malloc(num_frames * n_mels * sizeof(double));
    if (!melspec_buffer) {
        fprintf(stderr, "Failed to allocate memory for melspec buffer in compute_mfcc\n");
        return -1;
    }

    int frames_computed = compute_melspectrogram(params, audio_buffer, buffer_len, melspec_buffer);
    if (frames_computed <= 0) {
        free(melspec_buffer);
        return frames_computed;
    }

    for (int i = 0; i < frames_computed; ++i) {
        double* melspec_frame = melspec_buffer + (i * n_mels);
        double* mfcc_frame = mfcc_output + (i * n_mfcc);

        mfcc_power_to_db(melspec_frame, n_mels);
        dct2(melspec_frame, mfcc_frame, n_mels, n_mfcc);
    }

    free(melspec_buffer);
    return frames_computed;
}
