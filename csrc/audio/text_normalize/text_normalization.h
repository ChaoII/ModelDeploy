//
// Created by aichao on 2025/5/21.
//

#pragma once

#include <regex>
#include <filesystem>
#include <string>
#include <vector>

namespace modeldeploy::audio {
    class TextNormalizer {
    public:
        explicit TextNormalizer(const std::filesystem::path& char_map_folder);
        [[nodiscard]] std::vector<std::wstring> split(const std::wstring& text,
                                                      const std::wstring& lang = L"zh") const;
        static std::wstring post_replace(const std::wstring& sentence);
        std::wstring normalize_sentence(const std::wstring& sentence);
        std::vector<std::wstring> normalize(const std::wstring& text);

    private:
        std::wregex SENTENCE_SPLITOR;
    };
}
