//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/nv12_to_bgr.h"
#include "vision/common/processors/convert_and_permute.h"
#include "vision/common/processors/fusion_resize_pad_normalize_permute.h"
#include "vision/common/processors/hwc2chw.h"
#include "vision/utils.h"
#include "vision/face/face_det/scrfd_preproc.h"

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

bool CpuProcessorBackend::letterbox(const ImageData& image, ImageData* out,
                                    const std::vector<int>& dst_size,
                                    const std::vector<float>& padding_value,
                                    LetterBoxRecord* record) {
    cv::Mat mat;
    image.to_mat(mat);
    utils::letter_box(&mat, dst_size, padding_value, record);
    *out = ImageData(std::move(mat));
    return !out->empty();
}

bool CpuProcessorBackend::scrfd_preprocess(const ImageData& image, Tensor* out,
                                           const std::vector<int>& dst_size,
                                           float pad_val, LetterBoxRecord* record) {
    return scrfd_preprocess_cpu(image, out, dst_size, pad_val, record);
}

bool CpuProcessorBackend::resize(const ImageData& image, ImageData* out,
                                 int width, int height) {
    *out = image.resize(width, height);
    return !out->empty();
}

bool CpuProcessorBackend::convert(const ImageData& image, ImageData* out,
                                  const std::vector<float>& alpha,
                                  const std::vector<float>& beta) {
    *out = image.convert(alpha, beta);
    return !out->empty();
}

bool CpuProcessorBackend::cast(const ImageData& image, ImageData* out,
                               const std::string& dtype, bool scale) {
    *out = image.cast(dtype, scale);
    return !out->empty();
}

bool CpuProcessorBackend::convert_and_permute(const ImageData& image, Tensor* out,
                                              const std::vector<float>& alpha,
                                              const std::vector<float>& beta,
                                              bool swap_rb) {
    cv::Mat mat;
    image.to_mat(mat);
    if (!ConvertAndPermute::apply(&mat, alpha, beta, swap_rb)) return false;
    utils::mat_to_tensor(mat, out);
    return true;
}

bool CpuProcessorBackend::fusion_resize_pad_normalize_permute(
    const std::vector<ImageData>& images, Tensor* out,
    const std::vector<std::array<int, 2>>& resize_sizes,
    const std::vector<int>& dst_size,
    const std::vector<float>& mean, const std::vector<float>& std,
    float pad_value) {
    return fusion_resize_pad_normalize_permute_cpu(
        images, out, resize_sizes, dst_size, mean, std, pad_value);
}

bool CpuProcessorBackend::normalize(const ImageData& image, ImageData* out,
                                    const std::vector<float>& mean,
                                    const std::vector<float>& std,
                                    bool scale, bool swap_rb) {
    *out = image.normalize(mean, std, scale, swap_rb);
    return !out->empty();
}

bool CpuProcessorBackend::convert_to(const ImageData& image, ImageData* out,
                                     const std::string& dst_format) {
    ColorConvertType type;
    if (dst_format == "RGB") {
        type = ColorConvertType::CVT_PA_BGR2PA_RGB;
    } else if (dst_format == "GRAY") {
        type = ColorConvertType::CVT_PA_BGR2GRAY;
    } else if (dst_format == "BGR") {
        *out = image.clone();
        return !out->empty();
    } else {
        MD_LOG_ERROR << "Unsupported convert format: " << dst_format << std::endl;
        return false;
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
    cv::Mat mat;
    image.to_mat(mat);
    if (!HWC2CHW::apply(&mat)) return false;
    utils::mat_to_tensor(mat, out);
    return true;
}

bool CpuProcessorBackend::normalize_and_permute(const ImageData& image, Tensor* out,
                                                const std::vector<float>& mean,
                                                const std::vector<float>& std,
                                                bool scale) {
    auto tmp = image.fuse_normalize_and_permute(mean, std, scale);
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
