//
// Created by aichao on 2025/8/2.
//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {
    // 设备侧图像算子门控：决定 ImageData op 入口是否前置快速失败（不建 backend）。
    enum class ImageOp {
        Preprocess,
        Draw,
        Crop,
        Rotate,
        Resize,
        CvtColor,
        RotateCrop,
        PlaneSplit,
    };

    class MODELDEPLOY_CXX_EXPORT VisionProcessorBackend {
    public:
        // 是否支持在指定设备上执行某图像算子。
        // CPU→全支持；GPU→Preprocess/Draw/Crop（NV12 设备侧裁剪）已实现，其余中间 op 前置 fast-fail；
        // 设备帧(TPU)→放行中间算子(Crop/Rotate/CvtColor)由 SophgoProcessorBackend 就地实现，
        // Resize 未实现时仍返回 false（ImageData 置 last_error，不静默回退 CPU）。
        static bool supports(Device d, ImageOp op) {
            if (d == Device::CPU) return true;
            if (d == Device::GPU)
                return op == ImageOp::Preprocess || op == ImageOp::Draw ||
                       op == ImageOp::Crop;
            if (d == Device::TPU)
                return op == ImageOp::Preprocess || op == ImageOp::Draw ||
                       op == ImageOp::Crop || op == ImageOp::Rotate ||
                       op == ImageOp::CvtColor;
            return false;
        }

        virtual ~VisionProcessorBackend() = default;

        // YOLO 系融合算子（letterbox + resize + normalize + hwc2chw）
        virtual bool yolo_preprocess(const ImageData& image, Tensor* out,
                                     const std::vector<int>& dst_size,
                                     float pad_val, LetterBoxRecord* record) = 0;

        // NV12 直接输入（硬解码/摄像头常见格式）
        // src_device 指明 src_y/src_uv 所在内存的设备：CPU(默认)/GPU/TPU。
        virtual bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                          const std::vector<int>& src_size,
                                          int step_y, int step_uv, Tensor* out,
                                          const std::vector<int>& dst_size,
                                          float pad_val, LetterBoxRecord* record,
                                          Device src_device = Device::CPU) = 0;

        // 整批 yolo 预处理（batch>1 时一次 kernel 完成，避免 N 次 launch + concat）
        virtual bool yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                           const std::vector<int>& dst_size,
                                           float pad_val,
                                           std::vector<LetterBoxRecord>* records) = 0;

        // 整批通用融合预处理（batch>1 一次 kernel）：每图独立 origin/scale，共享 alpha/beta/swap_rb/pad
        // 输出 [batch, 3, dst_h, dst_w]（dst 尺寸 batch 内统一）
        virtual bool fused_preprocess_common_batch(
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

        // 整批 SCRFD 预处理（batch>1 一次 kernel 完成，避免 N 次 launch + concat）
        virtual bool scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                            const std::vector<int>& dst_size,
                                            float pad_val,
                                            std::vector<LetterBoxRecord>* records) = 0;

        // 通用算子（输出中间图像，供多算子 pipeline 串联）
        virtual bool resize(const ImageData& image, ImageData* out,
                            int width, int height) = 0;

        // 中间图像算子（crop/rotate/cvt_color）：由 ImageData 按 device() 分派到这里。
        // 默认返回 false（未实现后端 → ImageData 置 last_error，不静默回退 CPU）。
        virtual bool crop(const ImageData& image, float x, float y, float w, float h,
                          ImageData* out) {
            (void)image; (void)x; (void)y; (void)w; (void)h; (void)out;
            return false;
        }
        virtual bool rotate(const ImageData& image, RotateFlags flag, ImageData* out) {
            (void)image; (void)flag; (void)out;
            return false;
        }
        virtual bool cvt_color(const ImageData& image, ColorConvertType type, ImageData* out) {
            (void)image; (void)type; (void)out;
            return false;
        }
        virtual bool rotate_crop(const ImageData& image, std::array<float, 8> box, ImageData* out) {
            (void)image; (void)box; (void)out;
            return false;
        }

        // 整批融合算子（OCR det 用：resize+pad+normalize+permute，batch 内统一 pad）
        virtual bool ocr_det_preprocess(
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
            (void)device_image;
            (void)width;
            (void)height;
            (void)out;
            (void)record;
            return false;
        }

        // 通用融合预处理：crop/letterbox/resize -> bgr2rgb(可选) -> 仿射(alpha,beta) -> HWC2CHW
        // 一次完成，无中间缓冲（高效路径）。
        // 映射：src = (dst - origin) / scale；src 越界写 pad_value。
        //   - plain resize: origin=0, scale = dst/src
        //   - letterbox: origin = pad offset, scale = letterbox scale
        //   - center_crop: origin = crop origin, scale = dst/crop
        // alpha/beta 已合并 1/255 与 normalize：out_c = src_c * alpha[c] + beta[c]
        virtual bool fused_preprocess_common(
            const ImageData& image, Tensor* out,
            const std::vector<int>& dst_size,
            float origin_x, float origin_y,
            float scale_x, float scale_y,
            const std::vector<float>& alpha,
            const std::vector<float>& beta,
            bool swap_rb, float pad_value) = 0;

        // 双线性插值融合预处理：与 fused_preprocess_common 同映射，但采样用双线性插值。
        // 用于需要与 python cv2 INTER_LINEAR 对齐的场景（如 insightface）。
        virtual bool fused_preprocess_bilinear(
            const ImageData& image, Tensor* out,
            const std::vector<int>& dst_size,
            float origin_x, float origin_y,
            float scale_x, float scale_y,
            const std::vector<float>& alpha,
            const std::vector<float>& beta,
            bool swap_rb, float pad_value) {
            (void)image; (void)out; (void)dst_size;
            (void)origin_x; (void)origin_y; (void)scale_x; (void)scale_y;
            (void)alpha; (void)beta; (void)swap_rb; (void)pad_value;
            return false; // 默认不支持，由具体后端实现或走 CPU
        }

        // 通用融合预处理（颜色矩阵版）：采样/裁剪 + 3x3 颜色矩阵 + 偏置 + 写 CHW FP32。
        // 可表达 BGR2YCrCb / BGR2RGB 等任意 3x3 线性颜色变换 + 每通道偏置。
        // 默认返回 false（未实现后端可覆盖或走 CPU 兜底）。
        virtual bool fused_preprocess_color_matrix(
            const ImageData& image, Tensor* out,
            const std::vector<int>& dst_size,
            float origin_x, float origin_y,
            float scale_x, float scale_y,
            const float mat[3][3], const float bias[3],
            float pad_value) {
            (void)image; (void)out; (void)dst_size;
            (void)origin_x; (void)origin_y; (void)scale_x; (void)scale_y;
            (void)mat; (void)bias; (void)pad_value;
            return false;
        }

        // ── 设备侧绘制（NV12 帧就地绘制，保持 NV12）──
        // 坐标均为原图坐标（非 letterbox 空间）。返回 false 表示后端不支持/参数非法。
        // r/g/b 为 BGR 颜色分量（0-255）。thickness/radius 为像素。
        virtual bool draw_rect_nv12(ImageData& frame,
                                    float x, float y, float w, float h,
                                    float r, float g, float b, int thickness) {
            (void)frame; (void)x; (void)y; (void)w; (void)h;
            (void)r; (void)g; (void)b; (void)thickness;
            return false;
        }

        virtual bool draw_polygon_nv12(ImageData& frame,
                                       const std::vector<Point2f>& pts,
                                       float r, float g, float b, int thickness) {
            (void)frame; (void)pts; (void)r; (void)g; (void)b; (void)thickness;
            return false;
        }

        virtual bool draw_points_nv12(ImageData& frame,
                                      const std::vector<Point3f>& pts,
                                      float r, float g, float b, int radius) {
            (void)frame; (void)pts; (void)r; (void)g; (void)b; (void)radius;
            return false;
        }

        virtual bool draw_text_nv12(ImageData& frame,
                                    float x, float y, const std::string& text,
                                    float r, float g, float b, int font_size) {
            (void)frame; (void)x; (void)y; (void)text;
            (void)r; (void)g; (void)b; (void)font_size;
            return false;
        }

        // ── 设备侧高层可视化(设备 NV12 就地绘制,语义与 CPU vis_* 一致)──
        struct VisOptions {
            double threshold = 0.5;
            int font_size = 14;
            double alpha = 0.15;
            std::unordered_map<int, std::string> label_map;
            std::string font_path;   // 对 device 方法忽略
            bool save_result = false; // 对 device 方法忽略
        };

        virtual bool vis_det_nv12(ImageData& frame, const std::vector<DetectionResult>& result,
                                  const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_obb_nv12(ImageData& frame, const std::vector<ObbResult>& result,
                                  const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_pose_nv12(ImageData& frame, const std::vector<KeyPointsResult>& result,
                                   const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_keypoints_nv12(ImageData& frame, const std::vector<KeyPointsResult>& result,
                                        const VisOptions& opt, bool draw_lines) {
            (void)frame; (void)result; (void)opt; (void)draw_lines; return false;
        }
        virtual bool vis_hand_nv12(ImageData& frame, const std::vector<KeyPointsResult>& result,
                                   const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_ocr_nv12(ImageData& frame, const OCRResult& result, const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_lpr_nv12(ImageData& frame, const std::vector<LprResult>& result,
                                  const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_attr_nv12(ImageData& frame, const std::vector<AttributeResult>& result,
                                   const VisOptions& opt, const std::vector<int>& abnormal_ids,
                                   bool show_attr) {
            (void)frame; (void)result; (void)opt; (void)abnormal_ids; (void)show_attr; return false;
        }
        virtual bool vis_cls_nv12(ImageData& frame, const ClassifyResult& result,
                                  const VisOptions& opt, int top_k) {
            (void)frame; (void)result; (void)opt; (void)top_k; return false;
        }
        virtual bool vis_iseg_nv12(ImageData& frame, const std::vector<InstanceSegResult>& result,
                                   const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_sem_nv12(ImageData& frame, const SemSegResult& result, const VisOptions& opt) {
            (void)frame; (void)result; (void)opt; return false;
        }
        virtual bool vis_depth_nv12(ImageData& frame, const DepthResult& result,
                                    const VisOptions& opt, bool colorize) {
            (void)frame; (void)result; (void)opt; (void)colorize; return false;
        }
    };
} // namespace modeldeploy::vision
