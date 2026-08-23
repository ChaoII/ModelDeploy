#pragma once
#include <string>
#include "core/md_decl.h"
#include "audio/tools/wav_io.h"
namespace modeldeploy::audio::tool {
MODELDEPLOY_CXX_EXPORT WavMeta parse_meta(const std::string& path);
} // namespace modeldeploy::audio::tool
