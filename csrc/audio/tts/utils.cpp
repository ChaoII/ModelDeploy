//
// Created by aichao on 2025/5/22.
//

#include <fstream>
#include "csrc/audio/tts/utils.h"
#include "csrc/utils/utils.h"

namespace modeldeploy::audio::tts {
    std::vector<float> load_vec(const std::string& fin) {
        std::vector<float> ret;
        std::ifstream stream(fin);
        std::string line;
        while (std::getline(stream, line)) {
            auto vec = string_split(line, " ");
            for (const auto& s : vec) {
                ret.push_back(stof(s));
            }
        }
        return ret;
    }

    std::vector<std::string> utf8_to_charset(const std::string& input) {
        std::vector<std::string> output;
        for (size_t i = 0, len = 0; i != input.length(); i += len) {
            const unsigned char byte = static_cast<unsigned>(input[i]);
            if (byte >= 0xFC) // length 6
                len = 6;
            else if (byte >= 0xF8)
                len = 5;
            else if (byte >= 0xF0)
                len = 4;
            else if (byte >= 0xE0)
                len = 3;
            else if (byte >= 0xC0)
                len = 2;
            else
                len = 1;
            std::string ch = input.substr(i, len);
            output.push_back(ch);
        }
        return output;
    }

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
