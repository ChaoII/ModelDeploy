// csrc/audio/tts/common/ort_ep.h
#pragma once

#include <onnxruntime_cxx_api.h>

#include "core/enum_variables.h"  // Device
#include "core/md_decl.h"         // MODELDEPLOY_CXX_EXPORT

namespace modeldeploy::audio::tts {

// 若 dev==GPU 且本环境 ORT 提供 CUDA ExecutionProvider，则追加 CUDA EP 到 opts。
// 返回 true 表示已启用；false 表示未启用（device 非 GPU / provider 不可用 / 抛异常），
// 此时 opts 未被修改，调用方保持 CPU 执行即可（回退不失败）。
MODELDEPLOY_CXX_EXPORT bool ApplyOrtCudaEp(Ort::SessionOptions& opts, modeldeploy::Device dev,
                                           int device_id);

}  // namespace modeldeploy::audio::tts
