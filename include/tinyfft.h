#ifndef TINYFFT_H
#define TINYFFT_H

#include <complex.h>

// Define a type for complex numbers
typedef double complex cmplx;

/**
 * @brief Initializes the twiddle factors for FFT.
 * @param k The log2 of the FFT size (e.g., for 2048, k=11).
 * @param w A buffer to store the twiddle factors. Size must be 2^(k-1).
 */
void tfft_init(int k, cmplx *w);

/**
 * @brief Performs a forward Fast Fourier Transform.
 * @param k The log2 of the FFT size.
 * @param A The input/output buffer. Contains the time-domain signal on input,
 *          and the frequency-domain signal on output (in bit-reversed order).
 * @param w The precomputed twiddle factors.
 */
void tfft_fft(int k, cmplx *A, const cmplx *w);

/**
 * @brief Performs an inverse Fast Fourier Transform.
 * @param k The log2 of the FFT size.
 * @param A The input/output buffer. Input must be in bit-reversed order.
 * @param w The precomputed twiddle factors.
 */
void tfft_ifft(int k, cmplx *A, const cmplx *w);

/**
 * @brief Performs a cyclic convolution.
 */
void tfft_convolver(int k, cmplx *A, const cmplx *w);

#endif // TINYFFT_H
