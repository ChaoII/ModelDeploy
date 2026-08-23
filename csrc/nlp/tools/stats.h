#pragma once
#include <cstddef>
#include <string>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Stats {
    static size_t char_count(const std::string& text);
    static size_t word_count(const std::string& text);
    static size_t sentence_count(const std::string& text);
};
} // namespace modeldeploy::nlp::tool
