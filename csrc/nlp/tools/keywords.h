#pragma once
#include <string>
#include <utility>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Keywords {
    static std::vector<std::pair<std::string,int>> top(const std::string& text, int k = 5);
};
} // namespace modeldeploy::nlp::tool
