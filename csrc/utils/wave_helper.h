//
// Created by aichao on 2025/5/22.
//
#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "core/md_log.h"


struct WaveHeader {
    bool validate() const {
        //                 F F I R
        if (chunk_id != 0x46464952) {
            printf("Expected chunk_id RIFF. Given: 0x%08x\n", chunk_id);
            return false;
        }
        //               E V A W
        if (format != 0x45564157) {
            printf("Expected format WAVE. Given: 0x%08x\n", format);
            return false;
        }

        if (sub_chunk1_id != 0x20746d66) {
            printf("Expected subchunk1_id 0x20746d66. Given: 0x%08x\n", sub_chunk1_id);
            return false;
        }

        if (sub_chunk1_size != 16) {
            // 16 for PCM
            printf("Expected subchunk1_size 16. Given: %d\n", sub_chunk1_size);
            return false;
        }

        if (audio_format != 1) {
            // 1 for PCM
            printf("Expected audio_format 1. Given: %d\n", audio_format);
            return false;
        }

        if (num_channels != 1) {
            // we support only single channel for now
            printf("Expected single channel. Given: %d\n", num_channels);
            return false;
        }
        if (byte_rate != sample_rate * num_channels * bits_per_sample / 8) {
            return false;
        }

        if (block_align != num_channels * bits_per_sample / 8) {
            return false;
        }

        if (bits_per_sample != 16) {
            // we support only 16 bits per sample
            printf("Expected bits_per_sample 16. Given: %d\n", bits_per_sample);
            return false;
        }
        return true;
    }

    // See https://en.wikipedia.org/wiki/WAV#Metadata and
    // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
    void seek_to_data_chunk(std::istream& is) {
        //                              a t a d
        while (is && sub_chunk2_id != 0x61746164) {
            // const char *p = reinterpret_cast<const char *>(&subchunk2_id);
            // printf("Skip chunk (%x): %c%c%c%c of size: %d\n", subchunk2_id, p[0],
            //        p[1], p[2], p[3], subchunk2_size);
            is.seekg(sub_chunk2_size, std::istream::cur);
            is.read(reinterpret_cast<char*>(&sub_chunk2_id), sizeof(int32_t));
            is.read(reinterpret_cast<char*>(&sub_chunk2_size), sizeof(int32_t));
        }
    }

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
    int32_t sub_chunk2_size; // size of sub_chunk2
};

bool load_wav_file(const char* filename, int32_t* sampling_rate,
                   std::vector<float>& data) {
    WaveHeader header{};
    std::ifstream is(filename, std::ifstream::binary);
    is.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!is) {
        std::cout << "Failed to read " << filename;
        return false;
    }
    if (!header.validate()) {
        return false;
    }
    header.seek_to_data_chunk(is);
    if (!is) {
        return false;
    }

    *sampling_rate = header.sample_rate;
    // header.sub_chunk2_size contains the number of bytes in the data.
    // As we assume each sample contains two bytes, so it is divided by 2 here
    const auto speech_len = header.sub_chunk2_size / 2;
    data.resize(speech_len);

    auto speech_buff = (int16_t*)malloc(sizeof(int16_t) * speech_len);

    if (speech_buff) {
        memset(speech_buff, 0, sizeof(int16_t) * speech_len);
        is.read(reinterpret_cast<char*>(speech_buff), header.sub_chunk2_size);
        if (!is) {
            std::cout << "Failed to read " << filename;
            return false;
        }
        float scale = 32768;
        //float scale = 1.0;
        for (int32_t i = 0; i != speech_len; ++i) {
            data[i] = static_cast<float>(speech_buff[i]) / scale;
        }
        free(speech_buff);
        return true;
    }
    free(speech_buff);
    return false;
}
