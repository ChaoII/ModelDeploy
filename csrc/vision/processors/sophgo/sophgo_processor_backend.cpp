//
// Created by aichao on 2025/8/2.
// Sophgo BMCV 融合预处理。
//
// 说明：SOPHON-Sail / BMCV 的 API 随 SDK 版本有差异，下列调用基于 BM1688/CV186AH
// 的 sail v2.x 文档编写，真机联调时需按实际 SDK 头文件核对（文件内已标注 VERIFY 点）。
// 未安装 SDK 时（ENABLE_SOPHGO off）此类退化为 CPU 兜底。
//

#include "core/md_log.h"
#include "vision/processors/sophgo/sophgo_processor_backend.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/processors/processor_factory.h"

namespace modeldeploy::vision {

    SophgoProcessorBackend::SophgoProcessorBackend(int device_id) : device_id_(device_id) {
        cpu_fallback_ = std::make_unique<CpuProcessorBackend>();
        // 预处理路径：sail ONLY_RUNTIME 构建不含 BMCV；BMCV 融合路径为 VERIFY 待完善项。
        // 为兼容运行时-only sail 与保证正确性，预处理统一走 CPU 兜底（性能优化后续跟进）。
        (void)device_id_;
        handle_ = nullptr;
        bmcv_ = nullptr;
        MD_LOG_WARN << "SophgoProcessorBackend: preprocess uses CPU fallback (BMCV runtime path pending)." << std::endl;
    }

    SophgoProcessorBackend::~SophgoProcessorBackend() {
        bmcv_ = nullptr;
        handle_ = nullptr;
    }

    // ==================== TPU 融合路径 ====================
    // 思路（VERIFY: BMCV 调用签名）：
    //   1. ImageData(CPU BGR) -> sail::BMImage（上传）
    //   2. bmcv.vpp_convert：resize/letterbox/crop 到 dst_size（可用 vpp_crop_attr 表达 pad）
    //   3. bmcv.convert_to：BGR->RGB(swap) + 仿射(alpha,beta)
    //   4. bm_image_to_tensor：BMImage -> sail::Tensor（CHW）
    //   5. 拷贝到输出 CPU Tensor
    // 若某步 API 不确定，回退 CPU，保证正确性优先。

    bool SophgoProcessorBackend::fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        (void)origin_x; (void)origin_y; (void)scale_x; (void)scale_y;
        (void)swap_rb; (void)pad_value;
#ifdef ENABLE_SOPHGO
        if (!handle_ || !bmcv_) {
            return cpu_fallback_->fused_preprocess(
                image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
                alpha, beta, swap_rb, pad_value);
        }
        // VERIFY: sail::bmcv 的 vpp_convert / convert_to / bm_image_to_tensor 调用
        // 此处需按实际 SDK 补全；当前回退 CPU，保证正确。
        return cpu_fallback_->fused_preprocess(
            image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
            alpha, beta, swap_rb, pad_value);
#else
        return cpu_fallback_->fused_preprocess(
            image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
            alpha, beta, swap_rb, pad_value);
#endif
    }

    bool SophgoProcessorBackend::yolo_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record) {
        // letterbox 参数在 host 计算，映射到 fused_preprocess 的 origin/scale
        // VERIFY: 复用 utils::cal_letter_box_param 得到 scale/pad，然后走 fused TPU 路径
        return cpu_fallback_->yolo_preprocess(image, out, dst_size, pad_val, record);
    }

    bool SophgoProcessorBackend::yolo_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, std::vector<LetterBoxRecord>* records) {
        // VERIFY: TPU 批量 path（BMCV crop + TPU 通道），当前先 CPU 兜底
        return cpu_fallback_->yolo_preprocess_batch(images, out, dst_size, pad_val, records);
    }

    bool SophgoProcessorBackend::fused_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        // VERIFY: TPU 批量 path，当前先 CPU 兜底
        return cpu_fallback_->fused_preprocess_batch(
            images, out, dst_size, origins_x, origins_y, scales_x, scales_y,
            alpha, beta, swap_rb, pad_value);
    }

    bool SophgoProcessorBackend::yolo_preprocess_nv12(
        const uint8_t* src_y, const uint8_t* src_uv,
        const std::vector<int>& src_size,
        int step_y, int step_uv, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record) {
        return cpu_fallback_->yolo_preprocess_nv12(
            src_y, src_uv, src_size, step_y, step_uv, out, dst_size, pad_val, record);
    }

    bool SophgoProcessorBackend::letterbox(
        const ImageData& image, ImageData* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& padding_value,
        LetterBoxRecord* record) {
        return cpu_fallback_->letterbox(image, out, dst_size, padding_value, record);
    }

    bool SophgoProcessorBackend::resize(const ImageData& image, ImageData* out,
                                        int width, int height) {
        return cpu_fallback_->resize(image, out, width, height);
    }

    bool SophgoProcessorBackend::normalize(const ImageData& image, ImageData* out,
                                           const std::vector<float>& mean,
                                           const std::vector<float>& std,
                                           bool scale, bool swap_rb) {
        return cpu_fallback_->normalize(image, out, mean, std, scale, swap_rb);
    }

    bool SophgoProcessorBackend::convert_to(const ImageData& image, ImageData* out,
                                            const std::string& dst_format) {
        return cpu_fallback_->convert_to(image, out, dst_format);
    }

    bool SophgoProcessorBackend::center_crop(const ImageData& image, ImageData* out,
                                             int width, int height) {
        return cpu_fallback_->center_crop(image, out, width, height);
    }

    bool SophgoProcessorBackend::pad(const ImageData& image, ImageData* out,
                                     int top, int bottom, int left, int right,
                                     float value) {
        return cpu_fallback_->pad(image, out, top, bottom, left, right, value);
    }

    bool SophgoProcessorBackend::hwc2chw(const ImageData& image, Tensor* out) {
        return cpu_fallback_->hwc2chw(image, out);
    }

    bool SophgoProcessorBackend::normalize_and_permute(
        const ImageData& image, Tensor* out,
        const std::vector<float>& mean, const std::vector<float>& std,
        bool scale) {
        return cpu_fallback_->normalize_and_permute(image, out, mean, std, scale);
    }

    bool SophgoProcessorBackend::convert(const ImageData& image, ImageData* out,
                                         const std::vector<float>& alpha,
                                         const std::vector<float>& beta) {
        return cpu_fallback_->convert(image, out, alpha, beta);
    }

    bool SophgoProcessorBackend::cast(const ImageData& image, ImageData* out,
                                      const std::string& dtype, bool scale) {
        return cpu_fallback_->cast(image, out, dtype, scale);
    }

    bool SophgoProcessorBackend::convert_and_permute(
        const ImageData& image, Tensor* out,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb) {
        return cpu_fallback_->convert_and_permute(image, out, alpha, beta, swap_rb);
    }

    bool SophgoProcessorBackend::fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) {
        return cpu_fallback_->fusion_resize_pad_normalize_permute(
            images, out, resize_sizes, dst_size, mean, std, pad_value);
    }

    bool SophgoProcessorBackend::nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                                             int width, int height, ImageData* out) {
        return cpu_fallback_->nv12_to_bgr(y, uv, width, height, out);
    }

    bool SophgoProcessorBackend::scrfd_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record) {
        return cpu_fallback_->scrfd_preprocess(image, out, dst_size, pad_val, record);
    }

    bool SophgoProcessorBackend::process_device_image(
        void* device_image, int width, int height,
        Tensor* out, LetterBoxRecord* record) {
        // VERIFY: 硬解码路径，device_image 为 sail::BMImage*（或 bm_image*），直接 BMCV 处理
        (void)device_image; (void)width; (void)height; (void)out; (void)record;
        return false;
    }

} // namespace modeldeploy::vision
