//
// Created by aichao on 2025/8/2.
// Sophgo BMCV 融合预处理 + 设备内存零拷贝。
//
// 设计：SophgoProcessorBackend 继承 CpuProcessorBackend，仅覆写 yolo_preprocess /
// fused_preprocess_common / yolo_preprocess_nv12 三条 BMCV 硬件路径，其余算子自动回退 CPU。
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
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "bmlib_runtime.h"

namespace modeldeploy::vision {
    namespace {
        // 与 CPU/CUDA vis_* 确定性对齐的类调色板(RGB 序;20 色,label_id%20 取)
        constexpr uint8_t kClassPalette[20][3] = {
            {230, 159, 0}, {86, 180, 233}, {0, 158, 115}, {240, 228, 66},
            {0, 114, 178}, {213, 94, 0}, {204, 121, 167}, {255, 157, 0},
            {35, 105, 153}, {166, 86, 40}, {247, 129, 191}, {120, 87, 156},
            {255, 154, 161}, {107, 174, 214}, {222, 125, 44}, {152, 78, 163},
            {191, 119, 0}, {32, 74, 135}, {204, 154, 72}, {188, 143, 143}
        };
        static inline int palette_idx(int label_id) { return label_id >= 0 ? label_id % 20 : 0; }

        std::string label_name(const VisionProcessorBackend::VisOptions& opt, int label_id) {
            auto it = opt.label_map.find(label_id);
            if (it != opt.label_map.end() && !it->second.empty()) return it->second;
            return std::to_string(label_id);
        }
        std::string score_str(float score) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2f", score);
            return std::string(buf);
        }
        // bmcv_image_put_text 仅支持 ASCII 字库；非 ASCII(如中文 label/OCR/车牌)无法渲染。
        // 纯 ASCII 用原名，否则退化为 "id: score"；OCR/车牌文本过滤掉非 ASCII 可打印字符。
        static inline bool ascii_printable_only(const std::string& s) {
            for (unsigned char ch : s)
                if (ch < 0x20 || ch > 0x7E) return false;
            return true;
        }
        static inline std::string ascii_printable(const std::string& s) {
            std::string r;
            r.reserve(s.size());
            for (unsigned char ch : s)
                if (ch >= 0x20 && ch <= 0x7E) r.push_back(static_cast<char>(ch));
            return r;
        }
        // 提取可 BMCV 就地绘制的 TPU NV12 设备平面：返回 false 时不满足
        //（非 handle/非 TPU/非 NV12/平面不连续，均不得用宿主指针写设备显存 → 不回退 CPU）。
        static bool tpu_nv12_planes(void* handle, ImageData& frame,
                                    uint8_t** y, uint8_t** uv, int* w, int* h) {
            if (!handle || frame.device() != Device::TPU ||
                frame.type() != MdImageType::NV12 || frame.plane_count() < 2)
                return false;
            const auto p0 = frame.plane(0);
            const auto p1 = frame.plane(1);
            if (p0.step > 0 && p0.step != frame.width()) return false;
            if (p1.step > 0 && p1.step != frame.width()) return false;
            *y = const_cast<uint8_t*>(p0.data);
            *uv = const_cast<uint8_t*>(p1.data);
            *w = frame.width(); *h = frame.height();
            return true;
        }

        // COCO-17 骨架(1-indexed,画线时 -1)；pose 调色板(RGB 序)
        constexpr int kPoseSkeleton[19][2] = {
            {16,14},{14,12},{17,15},{15,13},{12,13},{6,12},{7,13},{6,7},{6,8},{7,9},
            {8,10},{9,11},{2,3},{1,2},{1,3},{2,4},{3,5},{4,6},{5,7}
        };
        constexpr uint8_t kPosePaletteRgb[20][3] = {
            {0,128,255},{51,153,255},{102,178,255},{0,230,230},{255,153,255},{255,204,153},
            {255,102,255},{255,51,255},{255,178,102},{255,153,51},{153,153,255},{102,102,255},
            {51,51,255},{153,255,153},{102,255,102},{51,255,51},{0,255,0},{255,0,0},
            {0,0,255},{255,255,255}
        };
        // MediaPipe hand 21 点骨架(0-indexed,20 条)与手部调色板
        constexpr int kHandSkeleton[20][2] = {
            {1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},
            {9,13},{13,14},{14,15},{15,16},{13,17},{17,18},{18,19},{19,20},{0,17}
        };
        constexpr uint8_t kHandPaletteRgb[21][3] = {
            {255,0,0},{255,85,0},{255,170,0},{255,255,0},{170,255,0},{85,255,0},
            {0,255,0},{0,255,85},{0,255,170},{0,255,255},{0,170,255},{0,85,255},
            {0,0,255},{85,0,255},{170,0,255},{255,0,255},{255,0,170},{255,0,85},
            {255,0,0},{170,0,0},{0,255,0}
        };

        // 批量绘制关键点与骨架：所有可见点聚合成一次 draw_points、所有线段聚合成一次 draw_lines（单色近似）。
        // 相比逐点/逐线多次 attach+draw，大幅降低对 BM1688 VPSS 通道的连续占用，避免绘制突发时的瞬时失败。
        static void draw_keypoints_batched(void* handle, uint8_t* y, uint8_t* uv, int W, int H,
                                           const std::vector<Point3f>& kpts,
                                           const int (*skeleton)[2], int nskeleton, bool one_indexed,
                                           const uint8_t* pts_color, const uint8_t* line_color) {
            std::vector<float> xs, ys;
            for (const auto& kp : kpts)
                if (kp.z >= 0.5f) { xs.push_back(kp.x); ys.push_back(kp.y); }
            if (!xs.empty())
                md_bmcv_draw_points_nv12(handle, y, uv, W, H, xs.data(), ys.data(),
                                         static_cast<int>(xs.size()), 3,
                                         pts_color[0], pts_color[1], pts_color[2]);
            if (!skeleton) return;
            std::vector<float> sx, sy, ex, ey;
            for (int j = 0; j < nskeleton; ++j) {
                int a = skeleton[j][0], b = skeleton[j][1];
                if (one_indexed) { --a; --b; }
                if (a < 0 || b < 0 || a >= (int)kpts.size() || b >= (int)kpts.size()) continue;
                if (kpts[a].z < 0.5f || kpts[b].z < 0.5f) continue;
                sx.push_back(kpts[a].x); sy.push_back(kpts[a].y);
                ex.push_back(kpts[b].x); ey.push_back(kpts[b].y);
            }
            if (!sx.empty())
                md_bmcv_draw_lines_nv12(handle, y, uv, W, H, sx.data(), sy.data(),
                                        ex.data(), ey.data(), static_cast<int>(sx.size()),
                                        line_color[0], line_color[1], line_color[2], 2);
        }
    } // namespace

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

    bool SophgoProcessorBackend::fused_preprocess_common(
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
                    MD_LOG_ERROR << "SophgoProcessorBackend: BMCV fused_preprocess_common failed (st="
                        << st << "), fallback to CPU." << std::endl;
                }
            }
        }
        return CpuProcessorBackend::fused_preprocess_common(
            image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
            alpha, beta, swap_rb, pad_value);
    }

    bool SophgoProcessorBackend::yolo_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float pad_val, LetterBoxRecord* record) {
        // letterbox 参数在 host 计算，映射到 fused_preprocess_common 的 origin/scale，走 BMCV 融合路径
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
        // CPU fused 的 pad_value 是输出空间(归一化后)；fused_preprocess_common 会按 alpha 还原
        return fused_preprocess_common(image, out, dst_size, pad_w, pad_h, scale, scale,
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

    // ── 设备侧高层可视化：BMCV 按能力近似实现（阶段 3）──
    // 说明：bmcv 无 alpha 半透明 → 填充为不透明；无任意多边形填充 → obb/OCR 仅外轮廓；
    // put_text 仅 ASCII → 中文/非 ASCII 文本退化为 id:score 或过滤；sem/depth 无 colormap → 返回 false。
    bool SophgoProcessorBackend::vis_det_nv12(ImageData& frame,
                                              const std::vector<DetectionResult>& result,
                                              const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
            std::string nm = label_name(opt, r.label_id);
            std::string label = (ascii_printable_only(nm) ? nm : std::to_string(r.label_id)) + ": " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(r.box.x), std::max(0.0f, r.box.y - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_obb_nv12(ImageData& frame,
                                              const std::vector<ObbResult>& result,
                                              const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            // 旋转矩形 → 4 角点（OpenCV RotatedRect 约定：angle 度数，绕中心逆时针）
            const double ang = r.rotated_box.angle * 3.14159265358979323846 / 180.0;
            const float cs = static_cast<float>(std::cos(ang));
            const float sn = static_cast<float>(std::sin(ang));
            const float hw = r.rotated_box.width * 0.5f, hh = r.rotated_box.height * 0.5f;
            const float cx = r.rotated_box.xc, cy = r.rotated_box.yc;
            float q[4][2] = { {-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh} };
            float xs[4], ys[4];
            for (int i = 0; i < 4; ++i) {
                xs[i] = cx + q[i][0] * cs - q[i][1] * sn;
                ys[i] = cy + q[i][0] * sn + q[i][1] * cs;
            }
            // bmcv 无任意多边形填充 → 仅外轮廓（近似，无内部填充）
            ok = (md_bmcv_draw_polygon_nv12(handle_, y, uv, W, H, xs, ys, 4, c[0], c[1], c[2], 2) == 0) && ok;
            std::string label = std::to_string(r.label_id) + ": " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(xs[1]), static_cast<int>(ys[1] - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_pose_nv12(ImageData& frame,
                                               const std::vector<KeyPointsResult>& result,
                                               const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(r.box.x), std::max(0.0f, r.box.y - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
            const auto& kpts = r.keypoints;
            // 批量绘制（聚合成单次 draw_points / draw_lines，单色近似，降低 VPSS 压力）
            draw_keypoints_batched(handle_, y, uv, W, H, kpts, kPoseSkeleton, 19, true,
                                   kPosePaletteRgb[16], kPosePaletteRgb[16]);
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_keypoints_nv12(ImageData& frame,
                                                    const std::vector<KeyPointsResult>& result,
                                                    const VisionProcessorBackend::VisOptions& opt,
                                                    bool draw_lines) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        // bmcv 下无骨架语义线条（与 CUDA 一致：未给 skeleton 时仅画点），draw_lines 无效
        (void)draw_lines;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(r.box.x), std::max(0.0f, r.box.y - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
            const auto& kpts = r.keypoints;
            // 仅点（无骨架），聚合为一次 draw_points（单色近似）
            draw_keypoints_batched(handle_, y, uv, W, H, kpts, nullptr, 0, false,
                                   kPosePaletteRgb[16], kPosePaletteRgb[16]);
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_hand_nv12(ImageData& frame,
                                               const std::vector<KeyPointsResult>& result,
                                               const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(r.box.x), std::max(0.0f, r.box.y - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
            const auto& kpts = r.keypoints;
            // 批量绘制（聚合点+骨架线段，单色近似，降低 VPSS 压力）
            draw_keypoints_batched(handle_, y, uv, W, H, kpts, kHandSkeleton, 20, false,
                                   kHandPaletteRgb[0], kHandPaletteRgb[0]);
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_ocr_nv12(ImageData& frame, const OCRResult& result,
                                              const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        (void)opt;
        bool ok = true;
        for (size_t i = 0; i < result.boxes.size(); ++i) {
            const auto& b = result.boxes[i];
            float xs[4], ys[4];
            for (int k = 0; k < 4; ++k) { xs[k] = static_cast<float>(b[k * 2]); ys[k] = static_cast<float>(b[k * 2 + 1]); }
            // bmcv 无任意多边形填充 → 仅外轮廓（近似）
            ok = (md_bmcv_draw_polygon_nv12(handle_, y, uv, W, H, xs, ys, 4, 66, 135, 245, 2) == 0) && ok;
            if (i < result.text.size() && !result.text[i].empty()) {
                std::string t = ascii_printable(result.text[i]);
                if (!t.empty())
                    ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                                 static_cast<int>(xs[0]), std::max(0, static_cast<int>(ys[0]) - 16),
                                                 t.c_str(), 255, 255, 255, 1) == 0) && ok;
            }
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_lpr_nv12(ImageData& frame,
                                              const std::vector<LprResult>& result,
                                              const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
            std::string label = ascii_printable(r.car_plate_str) + " " +
                                ascii_printable(r.car_plate_color) + " " + score_str(r.score);
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                         static_cast<int>(r.box.x), std::max(0.0f, r.box.y - 16.0f),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
            std::vector<float> xs, ys;
            for (const auto& p : r.keypoints) { xs.push_back(p.x); ys.push_back(p.y); }
            if (!xs.empty())
                ok = (md_bmcv_draw_points_nv12(handle_, y, uv, W, H, xs.data(), ys.data(),
                                               static_cast<int>(xs.size()), 3, 0, 255, 0) == 0) && ok;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_attr_nv12(ImageData& frame,
                                               const std::vector<AttributeResult>& result,
                                               const VisionProcessorBackend::VisOptions& opt,
                                               const std::vector<int>& abnormal_ids,
                                               bool show_attr) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        // 颜色语义对齐 CPU/CUDA：abnormal=红(255,0,0)、正常=绿(0,255,0)（RGB 序数组）
        std::unordered_map<int, bool> abnormal;
        for (int id : abnormal_ids) abnormal[id] = true;
        bool ok = true;
        int obj_idx = 0;
        for (const auto& r : result) {
            const bool is_abnormal = abnormal.count(obj_idx) > 0 && abnormal[obj_idx];
            uint8_t c[3];
            if (is_abnormal) { c[0] = 255; c[1] = 0; c[2] = 0; }
            else { c[0] = 0; c[1] = 255; c[2] = 0; }
            if (r.box_score >= opt.threshold) {
                ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H,
                                             static_cast<int>(r.box.x), static_cast<int>(r.box.y),
                                             static_cast<int>(r.box.x + r.box.width),
                                             static_cast<int>(r.box.y + r.box.height),
                                             c[0], c[1], c[2], 2) == 0) && ok;
                if (show_attr) {
                    for (size_t i = 0; i < r.attr_scores.size(); ++i) {
                        std::string nm = label_name(opt, static_cast<int>(i));
                        std::string a = (ascii_printable_only(nm) ? nm : std::to_string(i)) +
                                        ": " + score_str(r.attr_scores[i]);
                        const float ty = r.box.y + static_cast<float>((int)i + 1) * 16.0f;
                        ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H,
                                                     static_cast<int>(r.box.x), static_cast<int>(ty),
                                                     a.c_str(), 255, 255, 255, 1) == 0) && ok;
                    }
                }
            }
            ++obj_idx;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_cls_nv12(ImageData& frame, const ClassifyResult& result,
                                              const VisionProcessorBackend::VisOptions& opt,
                                              int top_k) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        if (top_k <= 0) top_k = 1;
        const int margin = 5;
        bool ok = true;
        int drawn = 0;
        const size_t n = std::min(result.label_ids.size(), result.scores.size());
        for (size_t i = 0; i < n && drawn < top_k; ++i) {
            if (result.scores[i] < opt.threshold) continue;
            const float yv = static_cast<float>(margin + drawn * 16);
            std::string label = std::to_string(result.label_ids[i]) + ": " + score_str(result.scores[i]);
            // bmcv 无半透明底块（fill 在 BM1688 不可用）→ 仅绘制文本行
            ok = (md_bmcv_draw_text_nv12(handle_, y, uv, W, H, margin, static_cast<int>(yv),
                                         label.c_str(), 255, 255, 255, 1) == 0) && ok;
            ++drawn;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_iseg_nv12(ImageData& frame,
                                               const std::vector<InstanceSegResult>& result,
                                               const VisionProcessorBackend::VisOptions& opt) {
        uint8_t* y = nullptr; uint8_t* uv = nullptr; int W = 0, H = 0;
        if (!tpu_nv12_planes(handle_, frame, &y, &uv, &W, &H)) return false;
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            // bmcv 无掩码叠加能力 → 近似为不透明框填充 + 外轮廓（不叠加实例 mask）
            ok = (md_bmcv_draw_rect_nv12(handle_, y, uv, W, H, x0, y0, x1, y1, c[0], c[1], c[2], 2) == 0) && ok;
        }
        return ok;
    }

    bool SophgoProcessorBackend::vis_sem_nv12(ImageData& frame, const SemSegResult& result,
                                              const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt;
        // bmcv 无 colormap/逐像素叠加能力 → 语义分割设备可视化不可用，返回 false。
        return false;
    }

    bool SophgoProcessorBackend::vis_depth_nv12(ImageData& frame, const DepthResult& result,
                                                const VisionProcessorBackend::VisOptions& opt,
                                                bool colorize) {
        (void)frame; (void)result; (void)opt; (void)colorize;
        // bmcv 无 colormap/逐像素叠加能力 → 深度设备可视化不可用，返回 false。
        return false;
    }
} // namespace modeldeploy::vision
