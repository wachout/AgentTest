#include "wav_utils.h"
#include "tinywav.h"
#include <stdlib.h>

int load_wav_file(const char* filename, WavData* wav_data) {
    if (!filename || !wav_data) {
        return -1;
    }

    TinyWav tw;
    // Ask tinywav to handle the channel splitting for us
    if (tinywav_open_read(&tw, filename, TW_SPLIT) != 0) {
        // Error message is printed by tinywav
        return -1;
    }

    wav_data->num_channels = tw.h.NumChannels;
    wav_data->sample_rate = tw.h.SampleRate;
    wav_data->num_samples = tw.numFramesInHeader;
    wav_data->audio_data = NULL;

    if (wav_data->num_samples == 0) {
        tinywav_close_read(&tw);
        return 0; // No data to read
    }

    // tinywav reads into a buffer of floats.
    // We need to allocate space for all channels to let tinywav do its thing.
    float* temp_buffer = (float*)malloc(wav_data->num_channels * wav_data->num_samples * sizeof(float));
    if (!temp_buffer) {
        fprintf(stderr, "Failed to allocate memory for temporary WAV buffer\n");
        tinywav_close_read(&tw);
        return -1;
    }

    // Create an array of pointers for tinywav to use for split channels
    float** channel_buffers = (float**)malloc(wav_data->num_channels * sizeof(float*));
    if (!channel_buffers) {
        fprintf(stderr, "Failed to allocate memory for channel pointers\n");
        free(temp_buffer);
        tinywav_close_read(&tw);
        return -1;
    }
    for (int i = 0; i < wav_data->num_channels; ++i) {
        channel_buffers[i] = temp_buffer + (i * wav_data->num_samples);
    }

    // Read the entire file
    int frames_read = tinywav_read_f(&tw, channel_buffers, wav_data->num_samples);

    // Allocate the final buffer for the first channel (double precision)
    wav_data->audio_data = (double*)malloc(frames_read * sizeof(double));
    if (!wav_data->audio_data) {
        fprintf(stderr, "Failed to allocate memory for final audio buffer\n");
        free(temp_buffer);
        free(channel_buffers);
        tinywav_close_read(&tw);
        return -1;
    }

    // Copy and convert the first channel from float to double
    float* first_channel = channel_buffers[0];
    for (int i = 0; i < frames_read; ++i) {
        wav_data->audio_data[i] = (double)first_channel[i];
    }

    // Update the number of samples in case we read fewer than the header claimed
    wav_data->num_samples = frames_read;

    // Clean up
    free(temp_buffer);
    free(channel_buffers);
    tinywav_close_read(&tw);

    return 0;
}

void free_wav_data(WavData* wav_data) {
    if (wav_data && wav_data->audio_data) {
        free(wav_data->audio_data);
        wav_data->audio_data = NULL;
    }
}
