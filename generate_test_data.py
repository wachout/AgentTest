import numpy as np
import librosa
import struct

def generate_data():
    """
    Generates a test audio signal and its corresponding MFCCs using librosa,
    saving both to binary files for C testing. Also saves intermediate
    Mel spectrogram for debugging.
    """
    # Parameters from the user request
    sr = 96000
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    n_mfcc = 13
    fmin = 0.0
    fmax = 48000.0
    duration = 1.0

    # Generate audio signal
    buffer_len = int(sr * duration)
    t = np.linspace(0., duration, buffer_len, endpoint=False)
    audio_signal = np.sin(2. * np.pi * 440. * t).astype(np.float64)
    print(f"Saving audio signal of length {len(audio_signal)} to test_signal.bin")
    audio_signal.tofile('test_signal.bin')

    # --- Intermediate Step: Mel Spectrogram ---
    # This is the output right after applying the Mel filterbank to the power spectrum.
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=False
    )
    # Transpose to (n_frames, n_mels) and save
    melspec_to_save = mel_spectrogram.T.astype(np.float64)
    print(f"Saving librosa Mel spectrogram of shape {melspec_to_save.shape} to librosa_melspec.bin")
    melspec_to_save.tofile('librosa_melspec.bin')

    # Print the first frame for debugging
    print("\n--- Librosa First Mel Spec Frame ---")
    for val in melspec_to_save[0, :5]:
        print(f"{val:.8f}", end=" ")
    print("...")


    # --- Final Step: MFCC ---
    # We can compute the MFCC from the pre-computed Mel spectrogram
    librosa_mfccs = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrogram, ref=np.max),
        n_mfcc=n_mfcc
    )

    # Transpose to (n_frames, n_mfcc) and save
    mfccs_to_save = librosa_mfccs.T.astype(np.float64)
    print(f"Saving librosa MFCCs of shape {mfccs_to_save.shape} to librosa_mfccs.bin")
    mfccs_to_save.tofile('librosa_mfccs.bin')

    print("Test data generation complete.")

if __name__ == '__main__':
    try:
        import librosa
        import numpy
    except ImportError:
        print("Error: librosa and numpy are required. Please install them using:")
        print("pip install librosa numpy")
        exit(1)

    generate_data()
