//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {

class MODELDEPLOY_CXX_EXPORT VisionProcessorBackend {
public:
    virtual ~VisionProcessorBackend() = default;

    // YOLO 系融合算子（letterbox + resize + normalize + hwc2chw）
    virtual bool yolo_preprocess(const ImageData& image, Tensor* out,
                                 const std::vector<int>& dst_size,
                                 float pad_val, LetterBoxRecord* record) = 0;

    // NV12 直接输入（硬解码/摄像头常见格式）
    virtual bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                      const std::vector<int>& src_size,
                                      int step_y, int step_uv, Tensor* out,
                                      const std::vector<int>& dst_size,
                                      float pad_val, LetterBoxRecord* record) = 0;

    // 纯 letterbox（不归一化不重排），输出 HWC 图像 + 记录（LPR det 用）
    virtual bool letterbox(const ImageData& image, ImageData* out,
                           const std::vector<int>& dst_size,
                           const std::vector<float>& padding_value,
                           LetterBoxRecord* record) = 0;

    // 整批 yolo 预处理（batch>1 时一次 kernel 完成，避免 N 次 launch + concat）
    virtual bool yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                       const std::vector<int>& dst_size,
                                       float pad_val,
                                       std::vector<LetterBoxRecord>* records) = 0;

    // 整批通用融合预处理（batch>1 一次 kernel）：每图独立 origin/scale，共享 alpha/beta/swap_rb/pad
    // 输出 [batch, 3, dst_h, dst_w]（dst 尺寸 batch 内统一）
    virtual bool fused_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) = 0;

    // SCRFD 人脸检测专用（letterbox + resize + normalize + hwc2chw，归一化为 (x-127.5)/128）
    virtual bool scrfd_preprocess(const ImageData& image, Tensor* out,
                                  const std::vector<int>& dst_size,
                                  float pad_val, LetterBoxRecord* record) = 0;

    // 通用算子（输出中间图像，供多算子 pipeline 串联）
    virtual bool resize(const ImageData& image, ImageData* out,
                        int width, int height) = 0;
    virtual bool normalize(const ImageData& image, ImageData* out,
                           const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool scale = true, bool swap_rb = true) = 0;
    virtual bool convert_to(const ImageData& image, ImageData* out,
                            const std::string& dst_format) = 0;

    // 数值缩放（如 /255），alpha/beta 与 Convert::apply 语义一致
    virtual bool convert(const ImageData& image, ImageData* out,
                         const std::vector<float>& alpha,
                         const std::vector<float>& beta) = 0;

    // 数据类型转换（如 uint8 -> float），dtype 与 Cast::apply 语义一致
    // scale=true 时乘以 1/255（与 ImageData::cast 默认一致），false 时纯类型转换
    virtual bool cast(const ImageData& image, ImageData* out,
                      const std::string& dtype, bool scale = true) = 0;

    // 缩放 + 通道重排（LPR 用：alpha=1/255, beta 可选, swap_rb）
    virtual bool convert_and_permute(const ImageData& image, Tensor* out,
                                     const std::vector<float>& alpha,
                                     const std::vector<float>& beta,
                                     bool swap_rb) = 0;

    virtual bool center_crop(const ImageData& image, ImageData* out,
                             int width, int height) = 0;
    // 四边填充，value 为填充值（OCR rec/cls/table 用）
    virtual bool pad(const ImageData& image, ImageData* out,
                     int top, int bottom, int left, int right,
                     float value = 0.0f) = 0;
    virtual bool hwc2chw(const ImageData& image, Tensor* out) = 0;
    virtual bool normalize_and_permute(const ImageData& image, Tensor* out,
                                       const std::vector<float>& mean,
                                       const std::vector<float>& std,
                                       bool scale = true) = 0;

    // 整批融合算子（OCR det 用：resize+pad+normalize+permute，batch 内统一 pad）
    virtual bool fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) = 0;
    virtual bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                             int width, int height, ImageData* out) = 0;

    // 零拷贝窄口子：直接吃设备侧图像（硬解码路径专用）
    virtual bool process_device_image(void* device_image, int width, int height,
                                      Tensor* out, LetterBoxRecord* record) {
        (void)device_image; (void)width; (void)height; (void)out; (void)record;
        return false;
    }

    // 通用融合预处理：crop/letterbox/resize -> bgr2rgb(可选) -> 仿射(alpha,beta) -> HWC2CHW
    // 一次完成，无中间缓冲（高效路径）。
    // 映射：src = (dst - origin) / scale；src 越界写 pad_value。
    //   - plain resize: origin=0, scale = dst/src
    //   - letterbox: origin = pad offset, scale = letterbox scale
    //   - center_crop: origin = crop origin, scale = dst/crop
    // alpha/beta 已合并 1/255 与 normalize：out_c = src_c * alpha[c] + beta[c]
    virtual bool fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) = 0;
};

} // namespace modeldeploy::vision
