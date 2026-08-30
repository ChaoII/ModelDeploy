#pragma once
// 仅测试用：GPU 设备可用性探测 + 优雅跳过工具。
// 所有需要真实 CUDA 设备的 [gpu] 用例在每个 TEST_CASE 开头调用 MD_TEST_GPU_OR_SKIP()：
// 无设备（本地无卡 / GitHub Actions 无设备 runner）时 WARN 并 return，不得 FAIL。
// CPU-only 构建（WITH_GPU 未定义）下该宏退化为空操作（return）。
#include <catch2/catch_test_macros.hpp>

#ifdef WITH_GPU

#include <cuda_runtime.h>

namespace modeldeploy::test_gpu {

inline bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace modeldeploy::test_gpu

#define MD_TEST_GPU_OR_SKIP()                                            \
    do {                                                                 \
        if (!::modeldeploy::test_gpu::available()) {                     \
            WARN("no CUDA device available; skipping GPU test at "       \
                 << __FILE__ << ":" << __LINE__);                        \
            return;                                                      \
        }                                                                \
    } while (0)

#else

#define MD_TEST_GPU_OR_SKIP() do { return; } while (0)

#endif  // WITH_GPU
