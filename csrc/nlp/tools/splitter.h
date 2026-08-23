#pragma once
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Splitter {
    static std::vector<std::string> split_sentences(const std::string& text);
};
} // namespace modeldeploy::nlp::tool
