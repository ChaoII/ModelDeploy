#pragma once

// 按 ModelConfig 构造 RuntimeOption 的唯一工厂：消除 inference_engine 与
// pipeline_manager 两处重复的后端/设备/TRT 缓存逻辑。
#include "config.hpp"
#include "csrc/runtime/runtime_option.h"

modeldeploy::RuntimeOption build_runtime_option(const ModelConfig& cfg);
