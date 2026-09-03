#pragma once

#include <cstddef>
#include <vector>
#include "core/tensor.h"

namespace modeldeploy {
namespace vision {
namespace ncnn_utils {

// ncnn 在 batch==1 时压掉输出首维。当 t 的秩 < expected_rank 时前置补 batch 维 1 到 expected_rank，
// 返回共享内存视图（Tensor::reshape 零拷贝）；秩达标时返回原 t（ORT/MNN 自动 no-op）。
// 仅用于 YOLO 家族后处理（输出 batch-first）。
inline Tensor restore_leading_batch1(const Tensor& t, const size_t expected_rank) {
    const size_t r = static_cast<size_t>(t.shape().size());
    if (r >= expected_rank) return t;
    std::vector<int64_t> shp(expected_rank, 1);
    for (size_t i = 0; i < r; ++i) {
        shp[expected_rank - r + i] = t.shape()[i];
    }
    return t.reshape(shp);
}

}  // namespace ncnn_utils
}  // namespace vision
}  // namespace modeldeploy
