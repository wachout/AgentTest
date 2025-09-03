import numpy as np
from scipy.io import wavfile

def generate_wav():
    """
    Generates a simple 16-bit PCM WAV file for testing.
    """
    sample_rate = 44100
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 note)

    # Generate a sine wave
    num_samples = int(sample_rate * duration)
    t = np.linspace(0., duration, num_samples, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5 # Use half of the max amplitude
    data = amplitude * np.sin(2. * np.pi * frequency * t)

    # Convert to 16-bit integer format
    wav_data = data.astype(np.int16)

    # Write to WAV file
    filename = "test.wav"
    wavfile.write(filename, sample_rate, wav_data)

    print(f"Successfully generated '{filename}'")
    print(f"  - Sample Rate: {sample_rate}")
    print(f"  - Duration: {duration}s")
    print(f"  - Format: 16-bit PCM")

if __name__ == '__main__':
    try:
        import numpy
        from scipy.io import wavfile
    except ImportError:
        print("Error: numpy and scipy are required. Please install them using:")
        print("pip install numpy scipy")
        exit(1)

    generate_wav()
