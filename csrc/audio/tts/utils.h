//
// Created by aichao on 2025/5/20.
//

#pragma once
#include "utils/utils.h"


namespace modeldeploy::audio::tts {
    std::vector<float> load_vec(const std::string& fin);

    std::vector<std::string> utf8_to_charset(const std::string& input);

    // see http://soundfile.sapp.org/doc/WaveFormat/
    //
    // Note: We assume little endian here
    // TODO: Support big endian
    struct WaveHeader {
        int32_t chunk_id;
        int32_t chunk_size;
        int32_t format;
        int32_t sub_chunk1_id;
        int32_t sub_chunk1_size;
        int16_t audio_format;
        int16_t num_channels;
        int32_t sample_rate;
        int32_t byte_rate;
        int16_t block_align;
        int16_t bits_per_sample;
        int32_t sub_chunk2_id; // a tag of this chunk
        int32_t sub_chunk2_size; // size of subchunk2
    };

    // Write a single channel wave file.
    // Note that the input samples are in the range [-1, 1]. It will be multiplied
    // by 32767 and saved in int16_t format in the wave file.
    //
    // @param filename Path to save the samples.
    // @param sampling_rate Sample rate of the samples.
    // @param samples Pointer to the samples
    // @param n Number of samples
    // @return Return true if the write succeeds; return false otherwise.
    MODELDEPLOY_CXX_EXPORT bool write_wave(const std::string& filename, int32_t sampling_rate,
                                           const float* samples, size_t n);

    MODELDEPLOY_CXX_EXPORT void write_wave(char* buffer, int32_t sampling_rate, const float* samples,
                                           size_t n);

    MODELDEPLOY_CXX_EXPORT size_t wave_file_size(int32_t n_samples);
}
