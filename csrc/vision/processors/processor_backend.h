//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {

class VisionProcessorBackend {
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

    // 通用算子（输出中间图像，供多算子 pipeline 串联）
    virtual bool resize(const ImageData& image, ImageData* out,
                        int width, int height) = 0;
    virtual bool normalize(const ImageData& image, ImageData* out,
                           const std::vector<float>& mean,
                           const std::vector<float>& std) = 0;
    virtual bool convert_to(const ImageData& image, ImageData* out,
                            const std::string& dst_format) = 0;
    virtual bool center_crop(const ImageData& image, ImageData* out,
                             int width, int height) = 0;
    virtual bool pad(const ImageData& image, ImageData* out,
                     const std::vector<int>& top,
                     const std::vector<int>& bottom) = 0;
    virtual bool hwc2chw(const ImageData& image, Tensor* out) = 0;
    virtual bool normalize_and_permute(const ImageData& image, Tensor* out,
                                       const std::vector<float>& mean,
                                       const std::vector<float>& std) = 0;
    virtual bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                             int width, int height, ImageData* out) = 0;

    // 零拷贝窄口子：直接吃设备侧图像（硬解码路径专用）
    virtual bool process_device_image(void* device_image, int width, int height,
                                      Tensor* out, LetterBoxRecord* record) {
        (void)device_image; (void)width; (void)height; (void)out; (void)record;
        return false;
    }
};

} // namespace modeldeploy::vision
