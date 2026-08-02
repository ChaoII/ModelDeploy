//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#ifdef WITH_GPU
#include "vision/processors/cuda/cuda_processor_backend.h"
#endif
#ifdef ENABLE_SOPHGO
#include "vision/processors/sophgo/sophgo_processor_backend.h"
#endif

namespace modeldeploy::vision {

std::unique_ptr<VisionProcessorBackend> create_processor_backend(
    Device device, Backend backend, int device_id) {
    if (device == Device::TPU && backend == Backend::SOPHGO) {
#ifdef ENABLE_SOPHGO
        return std::make_unique<SophgoProcessorBackend>(device_id);
#else
        MD_LOG_WARN << "SOPHGO not enabled, fallback to CPU processor backend." << std::endl;
        return std::make_unique<CpuProcessorBackend>();
#endif
    }
    switch (device) {
    case Device::GPU:
#ifdef WITH_GPU
        return std::make_unique<CudaProcessorBackend>();
#else
        MD_LOG_WARN << "GPU is not enabled, fallback to CPU processor backend." << std::endl;
        return std::make_unique<CpuProcessorBackend>();
#endif
    case Device::CPU:
    case Device::OPENCL:
    case Device::VULKAN:
    default:
        return std::make_unique<CpuProcessorBackend>();
    }
}

} // namespace modeldeploy::vision
