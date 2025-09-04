import numpy as np

def generate_stream_data():
    """
    Generates a simulated binary data file that mimics the user's
    voiceprint data stream format.
    NOTE: Using a smaller number of frames for testing to keep file size manageable.
    """
    NUM_FRAMES = 2 # Using 2 instead of 59 to keep file size small
    SAMPLES_PER_FRAME = 16384
    NUM_CHANNELS = 64
    FILENAME = "stream_data.bin"

    # Total samples for the first channel over 10 seconds
    total_samples_ch1 = NUM_FRAMES * SAMPLES_PER_FRAME

    # Create a sine wave for the first channel for the entire duration
    # This makes verification easier
    sample_rate = 96000 # Using the same SR as the original MFCC task
    duration = total_samples_ch1 / sample_rate
    frequency = 440.0
    t = np.linspace(0., duration, total_samples_ch1, endpoint=False)

    # Generate sine wave and scale to uint32 range
    # We'll scale it to use about half the dynamic range to be safe
    amplitude = (2**31 - 1) * 0.5
    sine_wave_float = amplitude * np.sin(2. * np.pi * frequency * t)

    # Shift to uint32 range
    # sine_wave_uint32 is a flat array of all samples for the first channel
    sine_wave_uint32 = (sine_wave_float + 2**31).astype(np.uint32)

    print(f"Generating '{FILENAME}' with {NUM_FRAMES} frames...")

    with open(FILENAME, "wb") as f:
        for i in range(NUM_FRAMES):
            # Create a buffer for one full frame (all channels)
            frame_buffer = np.zeros((SAMPLES_PER_FRAME, NUM_CHANNELS), dtype=np.uint32)

            # Get the segment of the sine wave for the current frame
            start_idx = i * SAMPLES_PER_FRAME
            end_idx = (i + 1) * SAMPLES_PER_FRAME

            # Place the sine wave data into the first channel
            frame_buffer[:, 0] = sine_wave_uint32[start_idx:end_idx]

            # Write the interleaved frame data to the file
            f.write(frame_buffer.tobytes())

    print("Data generation complete.")
    print(f"  - Frames: {NUM_FRAMES}")
    print(f"  - Samples per frame: {SAMPLES_PER_FRAME}")
    print(f"  - Channels: {NUM_CHANNELS}")
    print(f"  - Data type: uint32")

if __name__ == '__main__':
    generate_stream_data()
