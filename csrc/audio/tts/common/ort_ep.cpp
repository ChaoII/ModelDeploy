// csrc/audio/tts/common/ort_ep.cpp
#include "audio/tts/common/ort_ep.h"

#include <algorithm>
#include <memory>
#include <string>

#include "core/md_log.h"

namespace modeldeploy::audio::tts {

bool ApplyOrtCudaEp(Ort::SessionOptions& opts, modeldeploy::Device dev, int device_id) {
    if (dev != modeldeploy::Device::GPU) return false;
    // device_id 为 public 字段，可能绕过 set_device 直达负值，入口统一规约为 0
    if (device_id < 0) device_id = 0;

    try {
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

        OrtCUDAProviderOptionsV2* raw = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
        // RAII：任何抛出/成功路径都自动释放（Ort::detail::OrtRelease 即 ReleaseCUDAProviderOptions 包装）
        using CudaOptionsPtr =
            std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>;
        CudaOptionsPtr cuda_options(raw, &Ort::detail::OrtRelease);
        const std::string device_id_str = std::to_string(device_id);
        const char* keys[] = {"device_id"};
        const char* values[] = {device_id_str.c_str()};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
            cuda_options.get(), keys, values, static_cast<int>(sizeof(keys) / sizeof(keys[0]))));
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
            opts, cuda_options.get()));
    } catch (const std::exception& e) {
        MD_LOG_WARN << "OnnxRuntime CUDAExecutionProvider failed to enable: " << e.what()
                    << ". Fallback to CPU." << std::endl;
        return false;
    }
    MD_LOG_INFO << "OnnxRuntime CUDAExecutionProvider enabled for TTS." << std::endl;
    return true;
}

}  // namespace modeldeploy::audio::tts
