#pragma once
#include <string>
#include <optional>

namespace modeldeploy {
    struct NcnnBackendOption {
        int device_id = 0;
        int cpu_thread_num = -1;
        bool model_from_memory = false;
        std::string param_buffer;
        std::string bin_buffer;
        // ncnn Vulkan cooperative matrix 优化。某些 GPU/新版驱动（如 RTX 40 系 + 新驱动）
        // 上启用会触发 VK_ERROR_DEVICE_LOST，故默认关闭以保证稳定；在稳定驱动上可手动开启获得性能。
        bool use_cooperative_matrix = false;
        // ---- 常用 ncnn 运行参数；nullopt 表示不覆盖（沿用 ncnn 默认值）----
        // openmp 线程在休眠前忙等的时长（毫秒），-1 表示不覆盖
        int openmp_blocktime = -1;
        // 轻量模式：启用时回收中间 blob
        std::optional<bool> lightmode;
        // GPU/ARM 上的 fp16/bf16 精度路径
        std::optional<bool> use_fp16_packed;
        std::optional<bool> use_fp16_storage;
        std::optional<bool> use_fp16_arithmetic;
        std::optional<bool> use_bf16_storage;
    };
} // namespace modeldeploy
