#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    /// GPU 绘制盒：边框 + 半透明填充 + 顶部 8x16 标签条
    struct GpuDrawBox {
        int x1, y1, x2, y2;
        float score;
        int label_id;
        uint8_t r, g, b;   // BGR
        char label[32];
    };

    /// 在 BGR 图上绘制检测框（bgr/d_boxes 为设备指针；host 指针会自动上传并回拷）
    /// @return false 表示 CUDA 调用失败
    MODELDEPLOY_CXX_EXPORT bool draw_boxes_gpu(
        uint8_t* bgr, int width, int height,
        const GpuDrawBox* d_boxes, int num_boxes,
        float alpha = 0.15f, cudaStream_t stream = nullptr);
}
