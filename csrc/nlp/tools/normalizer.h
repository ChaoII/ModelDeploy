#pragma once
#include <string>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Normalizer {
    static std::string normalize(const std::string& text);
};
} // namespace modeldeploy::nlp::tool
