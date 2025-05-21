//
// Created by aichao on 2025/5/20.
//

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "csrc/audio/tts/wave_writer.h"

namespace modeldeploy::audio::tts {
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


    size_t wave_file_size(const int32_t n_samples) {
        return sizeof(WaveHeader) + n_samples * sizeof(int16_t);
    }

    void write_wave(char* buffer, const int32_t sampling_rate, const float* samples,
                    const size_t n) {
        WaveHeader header{};
        header.chunk_id = 0x46464952; // FFIR
        header.format = 0x45564157; // EVAW
        header.sub_chunk1_id = 0x20746d66; // "fmt "
        header.sub_chunk1_size = 16; // 16 for PCM
        header.audio_format = 1; // PCM =1

        constexpr int32_t num_channels = 1;
        constexpr int32_t bits_per_sample = 16; // int16_t
        header.num_channels = num_channels;
        header.sample_rate = sampling_rate;
        header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
        header.block_align = num_channels * bits_per_sample / 8;
        header.bits_per_sample = bits_per_sample;
        header.sub_chunk2_id = 0x61746164; // atad
        header.sub_chunk2_size = n * num_channels * bits_per_sample / 8;
        header.chunk_size = 36 + header.sub_chunk2_size;

        std::vector<int16_t> samples_int16(n);
        for (int32_t i = 0; i != n; ++i) {
            samples_int16[i] = static_cast<int16_t>(samples[i] * 32767);
        }
        memcpy(buffer, &header, sizeof(WaveHeader));
        memcpy(buffer + sizeof(WaveHeader), samples_int16.data(),
               n * sizeof(int16_t));
    }

    bool write_wave(const std::string& filename, const int32_t sampling_rate,
                    const float* samples, const size_t n) {
        std::string buffer;
        buffer.resize(wave_file_size(n));
        write_wave(buffer.data(), sampling_rate, samples, n);
        std::ofstream os(filename, std::ios::binary);
        if (!os) {
            return false;
        }
        os << buffer;
        if (!os) {
            return false;
        }
        return true;
    }
}
