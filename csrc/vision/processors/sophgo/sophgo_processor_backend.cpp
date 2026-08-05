//
// Created by aichao on 2025/8/2.
// Sophgo BMCV 融合预处理。
//
// 说明：SOPHON-Sail / BMCV 的 API 随 SDK 版本有差异，下列调用基于 BM1688/CV186AH
// 的 libsophon 0.4.x 编写。未安装 SDK 时（ENABLE_SOPHGO off）此类退化为 CPU 兜底。
//

#include "core/md_log.h"
#include "vision/processors/sophgo/sophgo_processor_backend.h"
#include "vision/processors/sophgo/bmcv_bridge.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/processors/processor_factory.h"
#include "vision/utils.h"
#include <algorithm>
#include <cmath>

#ifdef ENABLE_SOPHGO
#include "bmlib_runtime.h"
#endif

namespace modeldeploy::vision {

    SophgoProcessorBackend::SophgoProcessorBackend(int device_id) : device_id_(device_id) {
        cpu_fallback_ = std::make_unique<CpuProcessorBackend>();
        // handle 延迟初始化：不主动 bm_dev_request（避免与 bmrt 的 handle 冲突）。
        // 零拷贝路径由 UltralyticsDet 调用 use_external_handle(bmrt handle) 注入。
        // 普通 fused_preprocess（BMCV）在 handle 未设置时回退 CPU。
        handle_ = nullptr;
        MD_LOG_INFO << "SophgoProcessorBackend: BMCV handle deferred (set via use_external_handle)." << std::endl;
    }

    SophgoProcessorBackend::~SophgoProcessorBackend() {
#ifdef ENABLE_SOPHGO
        if (handle_ && !external_handle_) {
            bm_dev_free(static_cast<bm_handle_t>(handle_));
        }
        handle_ = nullptr;
#endif
    }

    void SophgoProcessorBackend::use_external_handle(void* handle) {
#ifdef ENABLE_SOPHGO
        if (handle_) {
            if (!external_handle_) {
                bm_dev_free(static_cast<bm_handle_t>(handle_));
            }
            handle_ = nullptr;
        }
        handle_ = handle;
        external_handle_ = true;
        MD_LOG_INFO << "SophgoProcessorBackend: using external bm_handle (shared with bmrt backend)." << std::endl;
#else
        (void)handle;
#endif
    }

    // ==================== TPU(BMCV) 融合路径 ====================
    // ImageData(BGR/RGB, HWC, uint8) -> 上传 bm_image -> vpp_convert_padding(letterbox/resize+pad)
    //   -> convert_to(BGR->RGB + alpha/beta 仿射, FP32) -> 读回 CPU Tensor(CHW)

    bool SophgoProcessorBackend::fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
#ifdef ENABLE_SOPHGO
        if (handle_ && dst_size.size() == 2 && alpha.size() == 3 && beta.size() == 3) {
            const int src_w = image.width();
            const int src_h = image.height();
            const int dst_w = dst_size[0];
            const int dst_h = dst_size[1];
            if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
                return cpu_fallback_->fused_preprocess(
                    image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
                    alpha, beta, swap_rb, pad_value);
            }

            // letterbox: src 全图 resize 到 (resize_w, resize_h)，放置于 dst 的 (pad_w, pad_h)
            int resize_w = static_cast<int>(std::lround(src_w * scale_x));
            int resize_h = static_cast<int>(std::lround(src_h * scale_y));
            int pad_w = static_cast<int>(std::lround(origin_x));
            int pad_h = static_cast<int>(std::lround(origin_y));
            resize_w = std::max(1, std::min(resize_w, dst_w));
            resize_h = std::max(1, std::min(resize_h, dst_h));
            pad_w = std::max(0, std::min(pad_w, dst_w - 1));
            pad_h = std::max(0, std::min(pad_h, dst_h - 1));

            // BMCV padding 是输入空间(0-255)；CPU fused 的 pad_value 是输出空间，按 alpha 还原
            const float scale0 = std::fabs(alpha[0]) > 1e-6f ? alpha[0] : 1.0f;
            int pad_raw = static_cast<int>(std::lround(pad_value / scale0));
            pad_raw = std::max(0, std::min(pad_raw, 255));
            const unsigned char p = static_cast<unsigned char>(pad_raw);

            out->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
            const int st = md_bmcv_letterbox_normalize(
                handle_, image.data(), src_w, src_h,
                out->data_ptr<float>(), dst_w, dst_h,
                pad_w, pad_h, resize_w, resize_h,
                alpha[0], alpha[1], alpha[2], p, swap_rb ? 1 : 0);
            if (st == 0) {
                out->expand_dim(0);
                return true;
            }
            MD_LOG_ERROR << "SophgoProcessorBackend: BMCV fused_preprocess failed (st="
                         << st << "), fallback to CPU." << std::endl;
        }
#endif
        return cpu_fallback_->fused_preprocess(
            image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
            alpha, beta, swap_rb, pad_value);
    }

    bool SophgoProcessorBackend::yolo_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record) {
        // letterbox 参数在 host 计算，映射到 fused_preprocess 的 origin/scale，走 BMCV 融合路径
        const float src_w = static_cast<float>(image.width());
        const float src_h = static_cast<float>(image.height());
        const float dst_w = static_cast<float>(dst_size[0]);
        const float dst_h = static_cast<float>(dst_size[1]);
        const float scale = std::min(dst_h / src_h, dst_w / src_w);
        const float resize_w = src_w * scale;
        const float resize_h = src_h * scale;
        const float pad_w = (dst_w - resize_w) * 0.5f;
        const float pad_h = (dst_h - resize_h) * 0.5f;
        *record = {src_w, src_h, dst_w, dst_h, pad_w, pad_h, scale};
        const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
        // CPU fused 的 pad_value 是输出空间(归一化后)；fused_preprocess 会按 alpha 还原
        return fused_preprocess(image, out, dst_size, pad_w, pad_h, scale, scale,
                                alpha, beta, true, pad_val / 255.0f);
    }

    bool SophgoProcessorBackend::yolo_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, std::vector<LetterBoxRecord>* records) {
        // VERIFY: TPU 批量 path（BMCV crop + TPU 通道），当前先 CPU 兜底
        return cpu_fallback_->yolo_preprocess_batch(images, out, dst_size, pad_val, records);
    }

    bool SophgoProcessorBackend::fused_preprocess_device(
        const ImageData& image, void** out_img, void* input_mem,
        int* dst_w, int* dst_h,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
#ifdef ENABLE_SOPHGO
        if (handle_ && out_img && input_mem && dst_w && dst_h &&
            alpha.size() == 3 && beta.size() == 3 && *dst_w > 0 && *dst_h > 0) {
            const int src_w = image.width();
            const int src_h = image.height();
            int resize_w = static_cast<int>(std::lround(src_w * scale_x));
            int resize_h = static_cast<int>(std::lround(src_h * scale_y));
            int pad_w = static_cast<int>(std::lround(origin_x));
            int pad_h = static_cast<int>(std::lround(origin_y));
            resize_w = std::max(1, std::min(resize_w, *dst_w));
            resize_h = std::max(1, std::min(resize_h, *dst_h));
            pad_w = std::max(0, std::min(pad_w, *dst_w - 1));
            pad_h = std::max(0, std::min(pad_h, *dst_h - 1));
            const float scale0 = std::fabs(alpha[0]) > 1e-6f ? alpha[0] : 1.0f;
            int pad_raw = static_cast<int>(std::lround(pad_value / scale0));
            pad_raw = std::max(0, std::min(pad_raw, 255));
            // 输出 bm_image（FP32 RGB_PLANAR），attach 到 input_mem（bmrt_tensor 分配的输入设备内存）
            void* oi = md_bmcv_image_create(handle_, *dst_w, *dst_h);
            if (!oi) {
                return false;
            }
            if (md_bmcv_letterbox_normalize_attach(
                    handle_, image.data(), src_w, src_h, oi, input_mem,
                    *dst_w, *dst_h, pad_w, pad_h, resize_w, resize_h,
                    alpha[0], alpha[1], alpha[2],
                    static_cast<unsigned char>(pad_raw), swap_rb ? 1 : 0) != 0) {
                md_bmcv_image_destroy(oi);
                MD_LOG_ERROR << "SophgoProcessorBackend: BMCV attach preprocess failed, fallback." << std::endl;
                return false;
            }
            *out_img = oi;
            return true;
        }
#endif
        return false;
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
        // VERIFY: 硬解码路径，device_image 为 bm_image*，直接 BMCV 处理（后续实现）
        (void)device_image; (void)width; (void)height; (void)out; (void)record;
        return false;
    }

} // namespace modeldeploy::vision
