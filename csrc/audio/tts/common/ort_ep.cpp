// csrc/audio/tts/common/ort_ep.cpp
#include "audio/tts/common/ort_ep.h"

#include <algorithm>
#include <string>
#include <vector>

#include "core/md_log.h"

namespace modeldeploy::audio::tts {

bool ApplyOrtCudaEp(Ort::SessionOptions& opts, modeldeploy::Device dev, int device_id) {
    if (dev != modeldeploy::Device::GPU) return false;

    // 仅当 provider 已注册才启用（Windows 下 provider DLL 须在可执行目录/搜索路径，
    // 参见 ort_backend.cpp 的同款检测；CPU 构建自然不含该项）。
    const auto providers = Ort::GetAvailableProviders();
    const bool has_cuda =
        std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
    if (!has_cuda) {
        MD_LOG_WARN << "OnnxRuntime CUDAExecutionProvider not available. Fallback to CPU."
                    << std::endl;
        return false;
    }

    try {
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
        const std::string device_id_str = std::to_string(device_id);
        const char* keys[] = {"device_id"};
        const char* values[] = {device_id_str.c_str()};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
            cuda_options, keys, values, static_cast<int>(sizeof(keys) / sizeof(keys[0]))));
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
            opts, cuda_options));
        Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    } catch (const std::exception& e) {
        MD_LOG_WARN << "OnnxRuntime CUDAExecutionProvider failed to enable: " << e.what()
                    << ". Fallback to CPU." << std::endl;
        return false;
    }
    MD_LOG_INFO << "OnnxRuntime CUDAExecutionProvider enabled for TTS." << std::endl;
    return true;
}

}  // namespace modeldeploy::audio::tts
