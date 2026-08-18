//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/processors/cpu/simd/fused_preproc_simd.h"
#include "vision/processors/cpu/yolo_preproc.h"
#include "vision/processors/cpu/nv12_to_bgr.h"
#include "vision/processors/cpu/fusion_resize_pad_normalize_permute.h"
#include "vision/processors/cpu/draw_nv12.h"
#include "vision/common/convert.h"
#include "vision/utils.h"
#include "vision/face/face_det/scrfd_preproc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace modeldeploy::vision {
    bool CpuProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                              const std::vector<int>& dst_size,
                                              float pad_val, LetterBoxRecord* record) {
        // 走 fused SIMD 通道
        *record = utils::cal_letter_box_param({image.width(), image.height()}, dst_size);
        float ox, oy, sx, sy;
        utils::letter_box_to_fused_params(*record, &ox, &oy, &sx, &sy);
        constexpr float alpha[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        constexpr float beta[3] = {0.0f, 0.0f, 0.0f};
        return fused_preprocess(image, out, dst_size, ox, oy, sx, sy,
                                std::vector<float>(alpha, alpha + 3),
                                std::vector<float>(beta, beta + 3),
                                true, pad_val / 255.0f);
    }

    bool CpuProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                                   const std::vector<int>& src_size,
                                                   int step_y, int step_uv, Tensor* out,
                                                   const std::vector<int>& dst_size,
                                                   float pad_val, LetterBoxRecord* record,
                                                   Device src_device) {
        if (src_device != Device::CPU) {
            MD_LOG_ERROR << "CpuProcessorBackend: NV12 src_device must be CPU, got "
                         << device_to_string(src_device) << "." << std::endl;
            return false;
        }
        return yolo_preprocess_nv12_cpu(src_y, src_uv, src_size, step_y, step_uv,
                                        out, dst_size, pad_val, record);
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

    bool CpuProcessorBackend::scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                     const std::vector<int>& dst_size,
                                                     float pad_val,
                                                     std::vector<LetterBoxRecord>* records) {
        if (images.empty() || dst_size.size() != 2) return false;
        const int batch = static_cast<int>(images.size());
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        records->resize(batch);
        std::vector<float> oxs(batch), oys(batch), sxs(batch), sys(batch);
        for (int i = 0; i < batch; ++i) {
            (*records)[i] = utils::cal_letter_box_param(
                {images[i].width(), images[i].height()}, {dst_w, dst_h});
            utils::letter_box_to_fused_params((*records)[i], &oxs[i], &oys[i], &sxs[i], &sys[i]);
        }
        const float alpha[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
        const float beta[3] = {-127.5f / 128.0f, -127.5f / 128.0f, -127.5f / 128.0f};
        return fused_preprocess_batch(images, out, dst_size,
                                      oxs, oys, sxs, sys,
                                      std::vector<float>(alpha, alpha + 3),
                                      std::vector<float>(beta, beta + 3),
                                      true, pad_val / 128.0f - 127.5f / 128.0f);
    }

    bool CpuProcessorBackend::resize(const ImageData& image, ImageData* out,
                                     int width, int height) {
        if (!out || width <= 0 || height <= 0) return false;
        cv::Mat src;
        if (!image.asMat(&src)) return false;
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(width, height));
        if (resized.empty()) return false;
        *out = ImageData(std::move(resized));
        return true;
    }

    bool CpuProcessorBackend::crop(const ImageData& image, float x, float y,
                                   float w, float h, ImageData* out) {
        if (!out || w <= 0 || h <= 0) return false;
        cv::Mat src;
        if (!image.asMat(&src)) return false;
        cv::Rect2f cv_rect(x, y, w, h);
        cv_rect = cv_rect & cv::Rect2f(0, 0, static_cast<float>(src.cols), static_cast<float>(src.rows));
        if (cv_rect.width <= 0 || cv_rect.height <= 0) return false;
        cv::Mat cropped = src(cv_rect).clone();
        if (cropped.empty()) return false;
        *out = ImageData(std::move(cropped));
        return true;
    }

    bool CpuProcessorBackend::rotate(const ImageData& image, RotateFlags flag, ImageData* out) {
        if (!out) return false;
        cv::Mat src;
        if (!image.asMat(&src)) return false;
        cv::Mat rotated;
        cv::rotate(src, rotated, static_cast<int>(flag));
        if (rotated.empty()) return false;
        *out = ImageData(std::move(rotated));
        return true;
    }

    bool CpuProcessorBackend::rotate_crop(const ImageData& image, std::array<float, 8> box, ImageData* out) {
        if (!out) return false;
        cv::Mat src;
        if (!image.asMat(&src)) return false;
        std::vector<std::vector<float>> points;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> tmp;
            tmp.push_back(box[2 * i]);
            tmp.push_back(box[2 * i + 1]);
            points.push_back(tmp);
        }
        float x_collect[4] = {box[0], box[2], box[4], box[6]};
        float y_collect[4] = {box[1], box[3], box[5], box[7]};
        float left = *std::min_element(x_collect, x_collect + 4);
        float right = *std::max_element(x_collect, x_collect + 4);
        float top = *std::min_element(y_collect, y_collect + 4);
        float bottom = *std::max_element(y_collect, y_collect + 4);
        cv::Rect roi(std::max(0, static_cast<int>(left)), std::max(0, static_cast<int>(top)),
                     std::max(1, static_cast<int>(right - left)), std::max(1, static_cast<int>(bottom - top)));
        cv::Mat img_crop;
        src(roi & cv::Rect(0, 0, src.cols, src.rows)).copyTo(img_crop);
        for (auto& point : points) {
            point[0] -= left;
            point[1] -= top;
        }

        const float img_crop_width = sqrt(pow(points[0][0] - points[1][0], 2) +
            pow(points[0][1] - points[1][1], 2));
        const float img_crop_height = sqrt(pow(points[0][0] - points[3][0], 2) +
            pow(points[0][1] - points[3][1], 2));

        cv::Point2f pts_std[4];
        pts_std[0] = cv::Point2f(0., 0.);
        pts_std[1] = cv::Point2f(img_crop_width, 0.);
        pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
        pts_std[3] = cv::Point2f(0.f, img_crop_height);

        cv::Point2f pointsf[4];
        pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
        pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
        pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
        pointsf[3] = cv::Point2f(points[3][0], points[3][1]);
        cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);
        cv::Mat dst_img;
        cv::warpPerspective(img_crop, dst_img, M,
                            cv::Size(img_crop_width, img_crop_height),
                            cv::BORDER_REPLICATE);

        if (dst_img.rows >= dst_img.cols * 1.5) {
            cv::transpose(dst_img, dst_img);
            cv::flip(dst_img, dst_img, 0);
        }
        *out = ImageData(dst_img);
        return !out->empty();
    }

    bool CpuProcessorBackend::cvt_color(const ImageData& image, ColorConvertType type, ImageData* out) {
        if (!out) return false;
        if (type == ColorConvertType::CVT_NV122PKG_BGR) {
            // NV12 → packed BGR（多平面，需 cvtColorTwoPlane，不能用单 mat asMat）
            if (image.format() != MdImageType::NV12) return false;
            const int w = image.width();
            const int h = image.height();
            if (w <= 0 || h <= 0 || (w % 2) != 0 || (h % 2) != 0) return false;
            const auto p0 = image.plane(0);
            const auto p1 = image.plane(1);
            if (!p0.data || !p1.data || p0.step <= 0 || p1.step <= 0) return false;
            cv::Mat y_mat(h, p0.step, CV_8UC1, const_cast<uint8_t*>(p0.data));
            cv::Mat uv_mat(h / 2, p1.step / 2, CV_8UC2, const_cast<uint8_t*>(p1.data));
            cv::Mat bgr;
            cv::cvtColorTwoPlane(y_mat(cv::Rect(0, 0, w, h)),
                                 uv_mat(cv::Rect(0, 0, w / 2, h / 2)),
                                 bgr, cv::COLOR_YUV2BGR_NV12);
            if (bgr.empty()) return false;
            *out = ImageData(std::move(bgr));
            return true;
        }
        if (type == ColorConvertType::CVT_I4202PKG_BGR) {
            // I420 → packed BGR（3 平面合并为单 buffer 再 cvtColor；不能用单 mat asMat）
            if (image.format() != MdImageType::I420) return false;
            const int w = image.width();
            const int h = image.height();
            if (w <= 0 || h <= 0 || (w % 2) != 0 || (h % 2) != 0 || image.plane_count() < 3) return false;
            const auto p0 = image.plane(0);
            const auto p1 = image.plane(1);
            const auto p2 = image.plane(2);
            if (!p0.data || !p1.data || !p2.data || p0.step <= 0 || p1.step <= 0 || p2.step <= 0) return false;
            const int uw = w / 2, uh = h / 2;
            std::vector<uint8_t> flat(static_cast<size_t>(w) * h * 3 / 2);
            uint8_t* d = flat.data();
            for (int r = 0; r < h; ++r)
                std::memcpy(d + static_cast<size_t>(r) * w,
                            p0.data + static_cast<size_t>(r) * p0.step, w);
            d += static_cast<size_t>(w) * h;
            for (int r = 0; r < uh; ++r)
                std::memcpy(d + static_cast<size_t>(r) * uw,
                            p1.data + static_cast<size_t>(r) * p1.step, uw);
            d += static_cast<size_t>(uw) * uh;
            for (int r = 0; r < uh; ++r)
                std::memcpy(d + static_cast<size_t>(r) * uw,
                            p2.data + static_cast<size_t>(r) * p2.step, uw);
            cv::Mat f(h * 3 / 2, w, CV_8UC1, flat.data());
            cv::Mat bgr;
            cv::cvtColor(f, bgr, cv::COLOR_YUV2BGR_I420);
            if (bgr.empty()) return false;
            if (!bgr.isContinuous()) bgr = bgr.clone();
            *out = ImageData(std::move(bgr));
            return true;
        }
        // PA（packed HWC）→ PL（planar CHW）：cv::split 拆通道后按平面平铺到连续 CHW 缓冲。
        if (type == ColorConvertType::CVT_PA_BGR2PL_BGR || type == ColorConvertType::CVT_PA_RGB2PL_RGB) {
            cv::Mat src;
            if (!image.asMat(&src)) return false;
            const int ch = src.channels();
            if (ch < 1) return false;
            std::vector<cv::Mat> chans;
            cv::split(src, chans);
            const MdImageType pl_type =
                (type == ColorConvertType::CVT_PA_BGR2PL_BGR)
                    ? (src.depth() == CV_8U ? MdImageType::PLA_BGR_U8 : MdImageType::PLA_BGR_F32)
                    : (src.depth() == CV_8U ? MdImageType::PLA_RGB_U8 : MdImageType::PLA_RGB_F32);
            const size_t plane_bytes =
                static_cast<size_t>(image.width()) * image.height() * static_cast<size_t>(src.elemSize1());
            std::vector<uint8_t> planar(plane_bytes * static_cast<size_t>(ch));
            uint8_t* pd = planar.data();
            for (int i = 0; i < ch; ++i) {
                if (!chans[i].isContinuous()) return false;
                std::memcpy(pd + static_cast<size_t>(i) * plane_bytes, chans[i].data, plane_bytes);
            }
            ImageData dst =
                ImageData::from_raw(planar.data(), image.width(), image.height(), pl_type, true, Device::CPU);
            if (dst.empty()) return false;
            *out = std::move(dst);
            return true;
        }
        // PL（planar CHW）→ PA（packed HWC）：按通道 rowRange 切出 H x W 视图后 cv::merge 交错。
        if (type == ColorConvertType::CVT_PL_BGR2PA_BGR || type == ColorConvertType::CVT_PL_RGB2PA_RGB) {
            if (!is_planar_type(image.format()) || image.plane_count() != 1) return false;
            const int ch = image.channels();
            const int w = image.width();
            const int h = image.height();
            const auto p0 = image.plane(0);
            if (!p0.data || ch < 1) return false;
            int ocv_depth = -1;
            switch (image.format()) {
            case MdImageType::PLA_BGR_U8: case MdImageType::PLA_RGB_U8:
            case MdImageType::PLA_BGRA_U8: case MdImageType::PLA_RGBA_U8:
                ocv_depth = CV_8U; break;
            case MdImageType::PLA_BGR_F32: case MdImageType::PLA_RGB_F32:
            case MdImageType::PLA_BGRA_F32: case MdImageType::PLA_RGBA_F32:
                ocv_depth = CV_32F; break;
            default:
                return false;
            }
            // 连续 CHW 平面 → [ch*h][w] 单通道视图，再按通道切出 H x W
            cv::Mat planar(h * ch, w, CV_MAKETYPE(ocv_depth, 1), const_cast<uint8_t*>(p0.data));
            std::vector<cv::Mat> chans(ch);
            for (int i = 0; i < ch; ++i)
                chans[i] = planar.rowRange(i * h, (i + 1) * h);
            cv::Mat hwc;
            cv::merge(chans, hwc);
            if (hwc.empty()) return false;
            *out = ImageData(std::move(hwc));
            return !out->empty();
        }
        const int ocv_type = md_color_convert_type_to_ocv_color_convert_type(type);
        if (ocv_type <= 0) return false;
        cv::Mat src;
        if (!image.asMat(&src)) return false;
        cv::Mat converted;
        cv::cvtColor(src, converted, ocv_type);
        if (converted.empty()) return false;
        *out = ImageData(std::move(converted));
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

    bool CpuProcessorBackend::nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                                          int width, int height, ImageData* out) {
        *out = ImageData(width, height, MdImageType::PKG_BGR_U8);
        if (out->empty()) return false;
        cv::Mat m;
        if (!out->asMat(&m)) return false;
        return nv12_to_bgr_cpu(y, uv, width, height, width, width, m.data);
    }

    bool CpuProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                    const std::vector<int>& dst_size,
                                                    float pad_val,
                                                    std::vector<LetterBoxRecord>* records) {
        if (images.empty() || dst_size.size() != 2) return false;
        const int batch = static_cast<int>(images.size());
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        records->resize(batch);
        // NV12 帧：先紧凑化 + 转 host BGR 视图，再走 packed BGR 批路径（CPU 正确性回退）
        std::vector<ImageData> imgs;
        std::vector<std::vector<uint8_t>> bgr_owns;
        imgs.reserve(batch);
        bgr_owns.reserve(batch);
        for (int i = 0; i < batch; ++i) {
            const ImageData& im = images[i];
            if (im.type() != MdImageType::NV12 || im.plane_count() < 2) {
                imgs.push_back(im);
                bgr_owns.emplace_back();
                continue;
            }
            if (im.device() != Device::CPU) {
                MD_LOG_ERROR << "CpuProcessorBackend::yolo_preprocess_batch: device NV12 frame requires GPU backend"
                             << std::endl;
                return false;
            }
            const int w = im.width(), h = im.height();
            const auto py = im.plane(0);
            const auto pu = im.plane(1);
            if (!py.data || !pu.data || w <= 0 || h <= 0) {
                MD_LOG_ERROR << "CpuProcessorBackend::yolo_preprocess_batch: invalid NV12 image #" << i
                             << std::endl;
                return false;
            }
            const int sy = py.step > 0 ? py.step : w;
            const int suv = pu.step > 0 ? pu.step : w;
            std::vector<uint8_t> nv12_buf(static_cast<size_t>(h) * w * 3 / 2);
            for (int r = 0; r < h; ++r)
                std::memcpy(nv12_buf.data() + static_cast<size_t>(r) * w, py.data + static_cast<size_t>(r) * sy, w);
            for (int r = 0; r < h / 2; ++r)
                std::memcpy(nv12_buf.data() + static_cast<size_t>(h) * w + static_cast<size_t>(r) * w,
                            pu.data + static_cast<size_t>(r) * suv, w);
            cv::Mat nv12_mat(h * 3 / 2, w, CV_8UC1, nv12_buf.data());
            auto& bgr_own = bgr_owns.emplace_back(static_cast<size_t>(w) * h * 3);
            cv::Mat bgr_mat(h, w, CV_8UC3, bgr_own.data());
            cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
            imgs.push_back(ImageData::from_raw(bgr_own.data(), w, h, MdImageType::PKG_BGR_U8, false));
        }
        // 整批一次遍历：每图独立 letterbox 映射，统一经 fused SIMD kernel 写入 batch 输出
        std::vector<float> oxs(batch), oys(batch), sxs(batch), sys(batch);
        for (int i = 0; i < batch; ++i) {
            (*records)[i] = utils::cal_letter_box_param(
                {imgs[i].width(), imgs[i].height()}, {dst_w, dst_h});
            utils::letter_box_to_fused_params((*records)[i], &oxs[i], &oys[i], &sxs[i], &sys[i]);
        }
        const float alpha[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const float beta[3] = {0.0f, 0.0f, 0.0f};
        return fused_preprocess_batch(imgs, out, dst_size,
                                      oxs, oys, sxs, sys,
                                      std::vector<float>(alpha, alpha + 3),
                                      std::vector<float>(beta, beta + 3),
                                      true, pad_val / 255.0f);
    }

    bool CpuProcessorBackend::fused_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        if (images.empty() || dst_size.size() != 2) return false;
        const int batch = static_cast<int>(images.size());
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        const int plane = 3 * dst_h * dst_w;
        out->allocate({batch, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = out->data_ptr<float>();
        const auto kernel = get_fused_preproc_kernel();
        for (int b = 0; b < batch; ++b) {
            kernel(images[b].plane(0).data, images[b].width(), images[b].height(),
                   dst + static_cast<size_t>(b) * plane, dst_w, dst_h,
                   origins_x[b], origins_y[b], scales_x[b], scales_y[b],
                   alpha.data(), beta.data(), swap_rb, pad_value);
        }
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
        const uint8_t* src = image.plane(0).data;

        // 直接 allocate 带 batch 维的 shape，避免 expand_dim 导致 shape 与下一帧不匹配而每帧重分配
        out->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = out->data_ptr<float>();

        // 运行时 ISA 派发（AVX512/AVX2/NEON/SVE/标量），一次遍历完成，无中间缓冲
        const auto kernel = get_fused_preproc_kernel();
        kernel(src, src_w, src_h, dst, dst_w, dst_h,
               origin_x, origin_y, scale_x, scale_y,
               alpha.data(), beta.data(), swap_rb, pad_value);
        return true;
    }

    bool CpuProcessorBackend::fused_preprocess_bilinear(
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
        const uint8_t* src = image.plane(0).data;

        out->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = out->data_ptr<float>();

        const auto kernel = get_fused_bilinear_preproc_kernel();
        kernel(src, src_w, src_h, dst, dst_w, dst_h,
               origin_x, origin_y, scale_x, scale_y,
               alpha.data(), beta.data(), swap_rb, pad_value);
        return true;
    }

    bool CpuProcessorBackend::fused_color_matrix_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const float mat[3][3], const float bias[3],
        float pad_value) {
        if (dst_size.size() != 2) return false;
        const int src_w = image.width();
        const int src_h = image.height();
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        const uint8_t* src = image.plane(0).data;

        out->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = out->data_ptr<float>();

        // 运行时 ISA 派发（颜色矩阵版），一次遍历完成
        const auto kernel = get_fused_color_matrix_kernel();
        kernel(src, src_w, src_h, dst, dst_w, dst_h,
               origin_x, origin_y, scale_x, scale_y,
                mat, bias, pad_value);
        return true;
    }

    bool CpuProcessorBackend::draw_rect_nv12(ImageData& frame, float x, float y, float w, float h,
                                             float r, float g, float b, int thickness) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_rect_nv12_cpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, w, h, static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), thickness);
    }

    bool CpuProcessorBackend::draw_polygon_nv12(ImageData& frame, const std::vector<Point2f>& pts,
                                                float r, float g, float b, int thickness) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_polygon_nv12_cpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                     frame.width(), frame.height(),
                                     pl0.step, pl1.step,
                                     pts, static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                     static_cast<uint8_t>(b), thickness);
    }

    bool CpuProcessorBackend::draw_points_nv12(ImageData& frame, const std::vector<Point3f>& pts,
                                               float r, float g, float b, int radius) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_points_nv12_cpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                    frame.width(), frame.height(),
                                    pl0.step, pl1.step,
                                    pts, static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                    static_cast<uint8_t>(b), radius);
    }

    bool CpuProcessorBackend::draw_text_nv12(ImageData& frame, float x, float y,
                                             const std::string& text,
                                             float r, float g, float b, int font_size) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_text_nv12_cpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, text, static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), font_size);
    }
} // namespace modeldeploy::vision
