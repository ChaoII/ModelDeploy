//
// Created by aichao on 2025/8/2.
// 通用融合预处理：center_crop/letterbox/plain-resize -> bgr2rgb(可选) -> 仿射(alpha,beta) -> HWC2CHW
// 一次 kernel launch 完成，无中间缓冲。CUDA nearest-neighbor，对齐 yolo_preproc.cu 的范式。
//
#pragma once

#include <cuda_runtime.h>
#include "core/tensor.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision {

// 通用融合预处理。
// @param src       [In] BGR 打包 uint8 图像指针（设备或主机内存均可，内部自动上传）
// @param src_size  [In] {w, h}
// @param out       [Out] 输出 Tensor，内部 allocate 为 {3, dst_h, dst_w} FP32 GPU CHW
// @param dst_size  [In] {w, h}
// @param origin_x, origin_y [In] 源图坐标映射偏移：src = (dst - origin) / scale
//         - plain resize: origin=0, scale = dst_size/src_size
//         - letterbox: origin = pad offset, scale = letterbox scale
//         - center_crop: origin = crop origin, scale = dst_size/crop_size
// @param scale_x, scale_y [In] 每轴缩放比
// @param alpha, beta [In] 每通道仿射：out_c = src_c * alpha[c] + beta[c]（已合并 1/255 与 normalize）
// @param swap_rb   [In] true 表示 BGR->RGB 通道交换（输出 C0=R）
// @param pad_value [In] 源坐标越界时的填充值（letterbox 或 OCR 右/下 pad）
// @param stream    [In] 可选 CUDA stream
bool fused_preprocess_cuda(const uint8_t* src,
                           const std::vector<int>& src_size,
                           Tensor* out,
                           const std::vector<int>& dst_size,
                           float origin_x, float origin_y,
                           float scale_x, float scale_y,
                           const std::vector<float>& alpha,
                           const std::vector<float>& beta,
                           bool swap_rb,
                           float pad_value,
                           cudaStream_t stream = nullptr);

// 整批通用融合预处理（3D grid 一次 launch）：每图独立 origin/scale，共享 alpha/beta/swap/pad
// 输出 [batch, 3, dst_h, dst_w] FP32 GPU，dst 尺寸 batch 内统一
bool fused_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                 Tensor* out,
                                 const std::vector<int>& dst_size,
                                 const std::vector<float>& origins_x,
                                 const std::vector<float>& origins_y,
                                 const std::vector<float>& scales_x,
                                 const std::vector<float>& scales_y,
                                 const std::vector<float>& alpha,
                                 const std::vector<float>& beta,
                                 bool swap_rb, float pad_value,
                                 cudaStream_t stream = nullptr);

} // namespace modeldeploy::vision
