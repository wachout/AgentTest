#include "mfcc.h"
#include "tinyfft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
static void bit_reversal_permutation(cmplx* buffer, int k);
unsigned int next_power_of_2(unsigned int n);
static int process_audio_stream(mfcc_params* params, const double* audio_buffer, size_t buffer_len,
                                double* power_spec_output, double* melspec_output, double* mfcc_output);


// --- Helper Functions ---
unsigned int next_power_of_2(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    n++;
    return n;
}
static double hz_to_mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }
static void hann_window(double* window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (length - 1)));
    }
}
static unsigned int reverse_bits(unsigned int num, unsigned int k) {
    unsigned int reversed_num = 0;
    for (unsigned int i = 0; i < k; i++) {
        if ((num >> i) & 1) reversed_num |= 1 << ((k - 1) - i);
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

// --- Core DSP Functions ---
void generate_mel_filterbank(double** filterbank, int n_mels, int n_fft, int sample_rate, double f_min, double f_max) {
    unsigned int fft_size = next_power_of_2(n_fft);
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
        int start_bin = fft_bins[i], center_bin = fft_bins[i + 1], end_bin = fft_bins[i + 2];
        for (int j = start_bin; j < center_bin; ++j) {
            if (j >= 0 && j < (int)(fft_size / 2 + 1)) filterbank[i][j] = (double)(j - start_bin) / (center_bin - start_bin) * enorm;
        }
        for (int j = center_bin; j < end_bin; ++j) {
            if (j >= 0 && j < (int)(fft_size / 2 + 1)) filterbank[i][j] = (double)(end_bin - j) / (end_bin - center_bin) * enorm;
        }
    }
    free(mel_points); free(hz_points); free(fft_bins);
}

void mfcc_power_to_db(double* spec, int len) {
    double amin = 1e-10, top_db = 80.0, max_val = 0.0, max_log_val = -1e100;
    for (int i = 0; i < len; ++i) if (spec[i] > max_val) max_val = spec[i];
    double ref_value = max_val;
    for (int i = 0; i < len; ++i) {
        double s_val = (spec[i] > amin) ? spec[i] : amin;
        double r_val = (ref_value > amin) ? ref_value : amin;
        spec[i] = 10.0 * (log10(s_val) - log10(r_val));
    }
    for (int i = 0; i < len; ++i) if (spec[i] > max_log_val) max_log_val = spec[i];
    for (int i = 0; i < len; ++i) if (spec[i] < max_log_val - top_db) spec[i] = max_log_val - top_db;
}

void dct2(const double* input, double* output, int num_mels, int num_mfcc) {
    for (int k = 0; k < num_mfcc; ++k) {
        double sum = 0.0;
        for (int n = 0; n < num_mels; ++n) sum += input[n] * cos(M_PI / num_mels * (n + 0.5) * k);
        double scale = (k == 0) ? sqrt(1.0 / num_mels) : sqrt(2.0 / num_mels);
        output[k] = sum * scale;
    }
}

// --- Internal Unified Processing Loop ---
static int process_audio_stream(mfcc_params* params, const double* audio_buffer, size_t buffer_len,
                                double* power_spec_output, double* melspec_output, double* mfcc_output) {
    // 1. Unpack parameters
    int n_fft = params->n_fft;
    int hop_length = params->hop_length;
    int n_mels = params->n_mels;
    int n_mfcc = params->n_mfcc;
    unsigned int fft_size = next_power_of_2(n_fft);
    unsigned int spec_len = fft_size / 2 + 1;

    if (buffer_len < (size_t)n_fft) return 0;
    int num_frames = 1 + (buffer_len - n_fft) / hop_length;

    // 2. Pre-computation and buffer allocation
    double* window = (double*)malloc(sizeof(double) * n_fft);
    hann_window(window, n_fft);

    double** mel_filterbank = (double**)malloc(sizeof(double*) * n_mels);
    for (int i = 0; i < n_mels; i++) {
        mel_filterbank[i] = (double*)calloc(spec_len, sizeof(double));
    }
    generate_mel_filterbank(mel_filterbank, n_mels, n_fft, params->sample_rate, params->fmin, params->fmax);

    int k = (int)log2(fft_size);
    cmplx* fft_twiddles = (cmplx*)malloc(sizeof(cmplx) * (1 << (k - 1)));
    tfft_init(k, fft_twiddles);

    // Per-frame buffers
    double* frame = (double*)malloc(sizeof(double) * n_fft);
    cmplx* fft_buffer = (cmplx*)malloc(sizeof(cmplx) * fft_size);
    double* power_spec_frame = (double*)malloc(sizeof(double) * spec_len);
    double* melspec_frame = (double*)malloc(sizeof(double) * n_mels);

    // 3. Main processing loop (frame by frame)
    for (int i = 0; i < num_frames; ++i) {
        // --- Framing and Windowing ---
        memcpy(frame, audio_buffer + (i * hop_length), sizeof(double) * n_fft);
        for (int j = 0; j < n_fft; ++j) frame[j] *= window[j];

        // --- FFT ---
        for (int j = 0; j < n_fft; ++j) fft_buffer[j] = frame[j] + 0.0 * I;
        for (unsigned int j = n_fft; j < fft_size; ++j) fft_buffer[j] = 0.0 + 0.0 * I;
        tfft_fft(k, fft_buffer, fft_twiddles);
        bit_reversal_permutation(fft_buffer, k);

        // --- Power Spectrogram ---
        for (unsigned int j = 0; j < spec_len; ++j) {
            power_spec_frame[j] = (creal(fft_buffer[j]) * creal(fft_buffer[j]) + cimag(fft_buffer[j]) * cimag(fft_buffer[j]));
        }
        if (power_spec_output) {
            memcpy(power_spec_output + (i * spec_len), power_spec_frame, spec_len * sizeof(double));
        }

        // --- Mel Spectrogram ---
        if (melspec_output || mfcc_output) {
            for (int j = 0; j < n_mels; ++j) {
                melspec_frame[j] = 0.0;
                for (unsigned int bin = 0; bin < spec_len; ++bin) {
                    melspec_frame[j] += power_spec_frame[bin] * mel_filterbank[j][bin];
                }
            }
            if (melspec_output) {
                memcpy(melspec_output + (i * n_mels), melspec_frame, n_mels * sizeof(double));
            }
        }

        // --- MFCC ---
        if (mfcc_output) {
            mfcc_power_to_db(melspec_frame, n_mels); // In-place
            dct2(melspec_frame, mfcc_output + (i * n_mfcc), n_mels, n_mfcc);
        }
    }

    // 4. Cleanup
    free(window);
    for (int i = 0; i < n_mels; i++) free(mel_filterbank[i]);
    free(mel_filterbank);
    free(fft_twiddles);
    free(frame);
    free(fft_buffer);
    free(power_spec_frame);
    free(melspec_frame);

    return num_frames;
}


// --- Public API Wrappers ---
int compute_power_spectrogram(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* spec_output) {
    return process_audio_stream(params, audio_buffer, buffer_len, spec_output, NULL, NULL);
}

int compute_melspectrogram(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* melspec_output) {
    return process_audio_stream(params, audio_buffer, buffer_len, NULL, melspec_output, NULL);
}

int compute_mfcc(mfcc_params* params, const double* audio_buffer, size_t buffer_len, double* mfcc_output) {
    return process_audio_stream(params, audio_buffer, buffer_len, NULL, NULL, mfcc_output);
}

int compute_features(mfcc_params* params, const double* audio_buffer, size_t buffer_len,
                     double* power_spec_output, double* melspec_output, double* mfcc_output) {
    return process_audio_stream(params, audio_buffer, buffer_len, power_spec_output, melspec_output, mfcc_output);
}
