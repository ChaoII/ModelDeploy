//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/processors/cpu/simd/fused_preproc_simd.h"
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
    // 走 fused SIMD 通道（标量参考 yolo_preprocess_cpu 仍保留）
    *record = utils::cal_letter_box_param({image.width(), image.height()}, dst_size);
    float ox, oy, sx, sy;
    utils::letter_box_to_fused_params(*record, &ox, &oy, &sx, &sy);
    const float alpha[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    const float beta[3] = {0.0f, 0.0f, 0.0f};
    return fused_preprocess(image, out, dst_size, ox, oy, sx, sy,
                            std::vector<float>(alpha, alpha + 3),
                            std::vector<float>(beta, beta + 3),
                            true, pad_val / 255.0f);
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
    // 走 fused SIMD 通道（scrfd 归一化 (x-127.5)/128）
    *record = utils::cal_letter_box_param({image.width(), image.height()}, dst_size);
    float ox, oy, sx, sy;
    utils::letter_box_to_fused_params(*record, &ox, &oy, &sx, &sy);
    const float alpha[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    const float beta[3] = {-127.5f / 128.0f, -127.5f / 128.0f, -127.5f / 128.0f};
    return fused_preprocess(image, out, dst_size, ox, oy, sx, sy,
                            std::vector<float>(alpha, alpha + 3),
                            std::vector<float>(beta, beta + 3),
                            true, pad_val / 128.0f - 127.5f / 128.0f);
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
                              int top, int bottom, int left, int right,
                              float value) {
    *out = image.pad(top, bottom, left, right, value);
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

bool CpuProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                const std::vector<int>& dst_size,
                                                float pad_val,
                                                std::vector<LetterBoxRecord>* records) {
    if (images.empty() || dst_size.size() != 2) return false;
    records->resize(images.size());
    std::vector<Tensor> tensors(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        if (!yolo_preprocess(images[i], &tensors[i], dst_size, pad_val, &(*records)[i])) return false;
    }
    *out = Tensor::concat(tensors, 0);
    return true;
}

bool CpuProcessorBackend::fused_preprocess(
    const ImageData& image, Tensor* out,
    const std::vector<int>& dst_size,
    float origin_x, float origin_y,
    float scale_x, float scale_y,
    const std::vector<float>& alpha,
    const std::vector<float>& beta,
    bool swap_rb, float pad_value) {
    if (dst_size.size() != 2 || alpha.size() != 3 || beta.size() != 3) return false;
    const int src_w = image.width();
    const int src_h = image.height();
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];
    const uint8_t* src = image.data();

    out->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
    float* dst = out->data_ptr<float>();

    // 运行时 ISA 派发（AVX512/AVX2/NEON/SVE/标量），一次遍历完成，无中间缓冲
    const auto kernel = get_fused_preproc_kernel();
    kernel(src, src_w, src_h, dst, dst_w, dst_h,
           origin_x, origin_y, scale_x, scale_y,
           alpha.data(), beta.data(), swap_rb, pad_value);
    out->expand_dim(0);
    return true;
}

} // namespace modeldeploy::vision
