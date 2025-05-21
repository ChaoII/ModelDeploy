//
// Created by aichao on 2025/5/20.
//

#pragma once
#include "csrc/utils/utils.h"


namespace modeldeploy::audio::tts {
    inline std::vector<float> load_vec(const std::string& fin) {
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

    inline std::vector<std::string> utf8_to_charset(const std::string& input) {
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
}
