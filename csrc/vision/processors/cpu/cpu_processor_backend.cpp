//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/nv12_to_bgr.h"

namespace modeldeploy::vision {

bool CpuProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                          const std::vector<int>& dst_size,
                                          float pad_val, LetterBoxRecord* record) {
    return yolo_preprocess_cpu(image, out, dst_size, pad_val, record);
}

bool CpuProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                               const std::vector<int>& src_size,
                                               int step_y, int step_uv, Tensor* out,
                                               const std::vector<int>& dst_size,
                                               float pad_val, LetterBoxRecord* record) {
    return yolo_preprocess_nv12_cpu(src_y, src_uv, src_size, step_y, step_uv,
                                    out, dst_size, pad_val, record);
}

bool CpuProcessorBackend::resize(const ImageData& image, ImageData* out,
                                 int width, int height) {
    *out = image.resize(width, height);
    return !out->empty();
}

bool CpuProcessorBackend::normalize(const ImageData& image, ImageData* out,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& std) {
    *out = image.normalize(mean, std);
    return !out->empty();
}

bool CpuProcessorBackend::convert_to(const ImageData& image, ImageData* out,
                                     const std::string& dst_format) {
    ColorConvertType type = ColorConvertType::CVT_PA_BGR2PA_RGB;
    if (dst_format == "RGB") {
        type = ColorConvertType::CVT_PA_BGR2PA_RGB;
    } else if (dst_format == "GRAY") {
        type = ColorConvertType::CVT_PA_BGR2GRAY;
    } else if (dst_format == "BGR") {
        *out = image.clone();
        return !out->empty();
    }
    *out = ImageData::cvt_color(image, type);
    return !out->empty();
}

bool CpuProcessorBackend::center_crop(const ImageData& image, ImageData* out,
                                      int width, int height) {
    *out = image.center_crop({width, height});
    return !out->empty();
}

bool CpuProcessorBackend::pad(const ImageData& image, ImageData* out,
                              const std::vector<int>& top,
                              const std::vector<int>& bottom) {
    *out = image.pad(top[0], bottom[0], top[1], bottom[1], 0.0f);
    return !out->empty();
}

bool CpuProcessorBackend::hwc2chw(const ImageData& image, Tensor* out) {
    ImageData tmp = image;
    tmp.to_tensor(out);
    return true;
}

bool CpuProcessorBackend::normalize_and_permute(const ImageData& image, Tensor* out,
                                                const std::vector<float>& mean,
                                                const std::vector<float>& std) {
    auto tmp = image.fuse_normalize_and_permute(mean, std);
    tmp.to_tensor(out);
    return true;
}

bool CpuProcessorBackend::nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                                      int width, int height, ImageData* out) {
    *out = ImageData(width, height, MdImageType::PKG_BGR_U8);
    if (out->empty()) return false;
    return nv12_to_bgr_cpu(y, uv, width, height, width, width, out->data());
}

} // namespace modeldeploy::vision
