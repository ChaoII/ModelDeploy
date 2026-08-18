//
// Created by aichao on 2025/8/2.
// Sophgo BMCV 融合预处理 + 设备内存零拷贝。
//
// 设计：SophgoProcessorBackend 继承 CpuProcessorBackend，仅覆写 yolo_preprocess /
// fused_preprocess / yolo_preprocess_nv12 三条 BMCV 硬件路径，其余算子自动回退 CPU。
// 硬件路径产出 Device::TPU Tensor（数据在自持的输入设备内存 in_mem_ 上），
// 交给 SophgoBackend::infer() 后识别 Device::TPU 输入跳过 H2D 直接 launch（零拷贝）。
// BMCV 调用失败或未编译 ENABLE_SOPHGO 时，回退到 CPU 实现（产出 CPU Tensor）。
//

#include "core/md_log.h"
#include "vision/processors/sophgo/sophgo_processor_backend.h"
#include "vision/processors/sophgo/bmcv_bridge.h"
#include "vision/utils.h"
#include <algorithm>
#include <cmath>

#include "bmlib_runtime.h"

namespace modeldeploy::vision {
    SophgoProcessorBackend::SophgoProcessorBackend(const int device_id) : device_id_(device_id) {
        bm_handle_t h = nullptr;
        if (bm_dev_request(&h, device_id_) == BM_SUCCESS) {
            handle_ = static_cast<void*>(h);
            MD_LOG_INFO << "SophgoProcessorBackend: BMCV handle ready (device " << device_id_ << ")." << std::endl;
        }
        else {
            MD_LOG_WARN << "SophgoProcessorBackend: bm_dev_request failed, BMCV disabled (CPU fallback)." << std::endl;
            handle_ = nullptr;
        }
    }

    SophgoProcessorBackend::~SophgoProcessorBackend() {
        if (handle_) {
            if (in_mem_) {
                bm_free_device(static_cast<bm_handle_t>(handle_),
                               *static_cast<bm_device_mem_t*>(in_mem_));
                delete[] static_cast<bm_device_mem_t*>(in_mem_);
                in_mem_ = nullptr;
            }
            bm_dev_free(static_cast<bm_handle_t>(handle_));
            handle_ = nullptr;
        }
    }

    void* SophgoProcessorBackend::ensure_input_mem(const int dst_w, const int dst_h) {
        if (!handle_ || dst_w <= 0 || dst_h <= 0) return nullptr;
        if (in_mem_ && cached_w_ == dst_w && cached_h_ == dst_h) {
            return in_mem_;
        }
        // 释放旧的并重新分配（尺寸变化时）
        if (in_mem_) {
            bm_free_device(static_cast<bm_handle_t>(handle_),
                           *static_cast<bm_device_mem_t*>(in_mem_));
            delete[] static_cast<bm_device_mem_t*>(in_mem_);
            in_mem_ = nullptr;
            in_mem_bytes_ = 0;
        }
        const size_t bytes = static_cast<size_t>(dst_h) * dst_w * 3 * sizeof(float);
        auto* mem = new bm_device_mem_t{};
        if (bm_malloc_device_byte(static_cast<bm_handle_t>(handle_), mem,
                                  static_cast<unsigned int>(bytes)) != BM_SUCCESS) {
            MD_LOG_ERROR << "SophgoProcessorBackend: bm_malloc_device_byte failed (" << bytes << " bytes)." <<
                std::endl;
            delete mem;
            return nullptr;
        }
        in_mem_ = mem;
        in_mem_bytes_ = bytes;
        cached_w_ = dst_w;
        cached_h_ = dst_h;
        return in_mem_;
    }

    bool SophgoProcessorBackend::finish_tpu_tensor(Tensor* out, const int dst_w, const int dst_h,
                                                   const std::string& name) {
        if (!in_mem_) return false;
        const std::vector<int64_t> shape = {1, 3, dst_h, dst_w};
        // data() 返回 bm_device_mem_t*，SophgoBackend::infer 识别 Device::TPU 输入直接 launch。
        // deleter 为空：设备内存由本 backend 持有并复用，Tensor 不拥有。
        out->from_external_memory(in_mem_, shape, DataType::FP32,
                                  [](void*) {
                                  }, Device::TPU, name);
        return true;
    }

    bool SophgoProcessorBackend::fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        if (handle_ && dst_size.size() == 2 && alpha.size() == 3 && beta.size() == 3) {
            const int src_w = image.width();
            const int src_h = image.height();
            const int dst_w = dst_size[0];
            const int dst_h = dst_size[1];
            if (src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0) {
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

                if (void* dev_mem = ensure_input_mem(dst_w, dst_h)) {
                    const int st = md_bmcv_letterbox_normalize_to_devmem(
                        handle_, image.plane(0).data, src_w, src_h, dev_mem, /* data migration (TPU CI verify) */
                        dst_w, dst_h, pad_w, pad_h, resize_w, resize_h,
                        alpha[0], alpha[1], alpha[2], p, swap_rb ? 1 : 0);
                    if (st == 0 && finish_tpu_tensor(out, dst_w, dst_h)) {
                        return true;
                    }
                    MD_LOG_ERROR << "SophgoProcessorBackend: BMCV fused_preprocess failed (st="
                        << st << "), fallback to CPU." << std::endl;
                }
            }
        }
        return CpuProcessorBackend::fused_preprocess(
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
        if (images.empty() || dst_size.size() != 2) return false;
        const int batch = static_cast<int>(images.size());
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        records->resize(batch);
        // 单图时直接走 yolo_preprocess（BMCV 单图零拷贝快路径）
        if (batch == 1) {
            return yolo_preprocess(images[0], out, dst_size, pad_val, &(*records)[0]);
        }
        // 逐图计算 letterbox 参数（host 侧，与单图一致）
        std::vector<float> oxs(batch), oys(batch), sxs(batch), sys(batch);
        for (int b = 0; b < batch; ++b) {
            const float src_w = static_cast<float>(images[b].width());
            const float src_h = static_cast<float>(images[b].height());
            const float scale = std::min(static_cast<float>(dst_h) / src_h,
                                         static_cast<float>(dst_w) / src_w);
            const float resize_w = src_w * scale;
            const float resize_h = src_h * scale;
            const float pad_w = (dst_w - resize_w) * 0.5f;
            const float pad_h = (dst_h - resize_h) * 0.5f;
            (*records)[b] = {src_w, src_h, static_cast<float>(dst_w),
                             static_cast<float>(dst_h), pad_w, pad_h, scale};
            oxs[b] = pad_w;
            oys[b] = pad_h;
            sxs[b] = scale;
            sys[b] = scale;
        }
        if (!handle_) {
            // 无 BMCV：回退 CPU batch
            return CpuProcessorBackend::yolo_preprocess_batch(images, out, dst_size, pad_val, records);
        }
        const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
        // BMCV 的 bm_image_attach 不支持任意偏移子视图，因此逐图用单图工作区处理，
        // 再 d2s 拷回 host 拼成 [batch,3,H,W] CPU tensor（行为与单图 BMCV 一致）。
        if (ensure_input_mem(dst_w, dst_h) != nullptr) {
            out->allocate({batch, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
            float* dst = out->data_ptr<float>();
            const size_t plane = static_cast<size_t>(dst_h) * dst_w * 3;
            bm_handle_t h = static_cast<bm_handle_t>(handle_);
            bm_device_mem_t& work = *static_cast<bm_device_mem_t*>(in_mem_);
            bool all_ok = true;
            for (int b = 0; b < batch; ++b) {
                const ImageData& image = images[b];
                const int src_w = image.width();
                const int src_h = image.height();
                int resize_w = static_cast<int>(std::lround(src_w * sxs[b]));
                int resize_h = static_cast<int>(std::lround(src_h * sys[b]));
                int pad_w = static_cast<int>(std::lround(oxs[b]));
                int pad_h = static_cast<int>(std::lround(oys[b]));
                resize_w = std::max(1, std::min(resize_w, dst_w));
                resize_h = std::max(1, std::min(resize_h, dst_h));
                pad_w = std::max(0, std::min(pad_w, dst_w - 1));
                pad_h = std::max(0, std::min(pad_h, dst_h - 1));
                // pad_val 是原始 0-255（与单图 yolo_preprocess 的输入一致），BMCV padding 直接使用
                int pad_raw = static_cast<int>(std::lround(pad_val));
                pad_raw = std::max(0, std::min(pad_raw, 255));
                const unsigned char p = static_cast<unsigned char>(pad_raw);
                const int st = md_bmcv_letterbox_normalize_to_devmem(
                    handle_, image.plane(0).data, src_w, src_h, &work, /* data migration (TPU CI verify) */
                    dst_w, dst_h, pad_w, pad_h, resize_w, resize_h,
                    alpha[0], alpha[1], alpha[2], p, /*swap_rb*/1);
                if (st != 0) {
                    all_ok = false;
                    break;
                }
                if (bm_memcpy_d2s(h, dst + b * plane, work) != BM_SUCCESS) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                return true;
            }
            MD_LOG_ERROR << "SophgoProcessorBackend: BMCV yolo_preprocess_batch failed, fallback to CPU." << std::endl;
        }
        return CpuProcessorBackend::yolo_preprocess_batch(images, out, dst_size, pad_val, records);
    }

    bool SophgoProcessorBackend::yolo_preprocess_nv12(
        const uint8_t* src_y, const uint8_t* src_uv,
        const std::vector<int>& src_size,
        int step_y, int step_uv, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record,
        Device src_device) {
        if (src_device == Device::GPU) {
            MD_LOG_ERROR << "SophgoProcessorBackend: NV12 src_device GPU not supported." << std::endl;
            return false;
        }
        if (handle_ && src_y && src_uv && src_size.size() == 2 && dst_size.size() == 2) {
            const int src_w = src_size[0];
            const int src_h = src_size[1];
            const int dst_w = dst_size[0];
            const int dst_h = dst_size[1];
            if (src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0) {
                *record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
                const float scale0 = 1.0f / 255.0f;
                int pad_raw = static_cast<int>(std::lround(pad_val));
                pad_raw = std::max(0, std::min(pad_raw, 255));
                const unsigned char p = static_cast<unsigned char>(pad_raw);

                if (void* dev_mem = ensure_input_mem(dst_w, dst_h)) {
                    const int st = md_bmcv_nv12_letterbox_normalize_to_devmem(
                        handle_, src_y, src_uv, src_w, src_h,
                        step_y, step_uv, dev_mem, dst_w, dst_h,
                        scale0, scale0, scale0, p,
                        src_device == Device::TPU);
                    if (st == 0 && finish_tpu_tensor(out, dst_w, dst_h)) {
                        return true;
                    }
                    MD_LOG_ERROR << "SophgoProcessorBackend: BMCV NV12 preprocess failed (st="
                        << st << "), fallback to CPU." << std::endl;
                }
            }
        }
        return CpuProcessorBackend::yolo_preprocess_nv12(
            src_y, src_uv, src_size, step_y, step_uv, out, dst_size, pad_val, record, src_device);
    }

    // NV12 设备绘制辅助：仅当 frame 为 TPU 设备帧且 Y/UV 平面步长与宽度连续时走 BMCV 设备路径；
    // 否则返回 false（不回退 CPU，因为 CPU 顺序实现会用宿主指针写设备显存，属 UB）。
    bool SophgoProcessorBackend::draw_rect_nv12(ImageData& frame, float x, float y, float w, float h,
                                                float r, float g, float b, int thickness) {
        // plane migration (TPU CI verify)
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        if (handle_ && frame.device() == Device::TPU &&
            frame.type() == MdImageType::NV12 &&
            !(pl0.step > 0 && pl0.step != frame.width()) &&
            !(pl1.step > 0 && pl1.step != frame.width())) {
            const int st = md_bmcv_draw_rect_nv12(
                handle_, const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                frame.width(), frame.height(),
                static_cast<int>(x), static_cast<int>(y),
                static_cast<int>(x + w), static_cast<int>(y + h),
                static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), thickness);
            if (st == 0) return true;
            MD_LOG_WARN << "SophgoProcessorBackend: BMCV draw_rect_nv12 failed (st=" << st << ")." << std::endl;
        }
        return false;
    }

    bool SophgoProcessorBackend::draw_polygon_nv12(ImageData& frame, const std::vector<Point2f>& pts,
                                                   float r, float g, float b, int thickness) {
        // plane migration (TPU CI verify)
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        if (handle_ && frame.device() == Device::TPU &&
            frame.type() == MdImageType::NV12 &&
            !(pl0.step > 0 && pl0.step != frame.width()) &&
            !(pl1.step > 0 && pl1.step != frame.width()) &&
            pts.size() >= 3) {
            std::vector<float> xs, ys;
            xs.reserve(pts.size()); ys.reserve(pts.size());
            for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
            const int st = md_bmcv_draw_polygon_nv12(
                handle_, const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                frame.width(), frame.height(), xs.data(), ys.data(), static_cast<int>(xs.size()),
                static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), thickness);
            if (st == 0) return true;
            MD_LOG_WARN << "SophgoProcessorBackend: BMCV draw_polygon_nv12 failed (st=" << st << ")." << std::endl;
        }
        return false;
    }

    bool SophgoProcessorBackend::draw_points_nv12(ImageData& frame, const std::vector<Point3f>& pts,
                                                  float r, float g, float b, int radius) {
        // plane migration (TPU CI verify)
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        if (handle_ && frame.device() == Device::TPU &&
            frame.type() == MdImageType::NV12 &&
            !(pl0.step > 0 && pl0.step != frame.width()) &&
            !(pl1.step > 0 && pl1.step != frame.width()) &&
            !pts.empty()) {
            std::vector<float> xs, ys;
            xs.reserve(pts.size()); ys.reserve(pts.size());
            for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
            const int st = md_bmcv_draw_points_nv12(
                handle_, const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                frame.width(), frame.height(), xs.data(), ys.data(), static_cast<int>(xs.size()),
                radius, static_cast<int>(r), static_cast<int>(g), static_cast<int>(b));
            if (st == 0) return true;
            MD_LOG_WARN << "SophgoProcessorBackend: BMCV draw_points_nv12 failed (st=" << st << ")." << std::endl;
        }
        return false;
    }

    bool SophgoProcessorBackend::draw_text_nv12(ImageData& frame, float x, float y,
                                                const std::string& text,
                                                float r, float g, float b, int font_size) {
        // plane migration (TPU CI verify)
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        if (handle_ && frame.device() == Device::TPU &&
            frame.type() == MdImageType::NV12 &&
            !(pl0.step > 0 && pl0.step != frame.width()) &&
            !(pl1.step > 0 && pl1.step != frame.width()) &&
            !text.empty()) {
            const int st = md_bmcv_draw_text_nv12(
                handle_, const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                frame.width(), frame.height(),
                static_cast<int>(x), static_cast<int>(y), text.c_str(),
                static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), font_size);
            if (st == 0) return true;
            MD_LOG_WARN << "SophgoProcessorBackend: BMCV draw_text_nv12 failed (st=" << st << ")." << std::endl;
        }
        return false;
    }
} // namespace modeldeploy::vision
