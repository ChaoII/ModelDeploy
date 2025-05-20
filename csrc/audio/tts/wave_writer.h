//
// Created by aichao on 2025/5/20.
//

#pragma once

#include <cstdint>
#include <string>

namespace modeldeploy {
    // Write a single channel wave file.
    // Note that the input samples are in the range [-1, 1]. It will be multiplied
    // by 32767 and saved in int16_t format in the wave file.
    //
    // @param filename Path to save the samples.
    // @param sampling_rate Sample rate of the samples.
    // @param samples Pointer to the samples
    // @param n Number of samples
    // @return Return true if the write succeeds; return false otherwise.
    bool write_wave(const std::string& filename, int32_t sampling_rate,
                   const float* samples, size_t n);

    void write_wave(char* buffer, int32_t sampling_rate, const float* samples,
                   size_t n);

    size_t wave_file_size(int32_t n_samples);
}
