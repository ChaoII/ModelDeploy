//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#include "vision/processors/cuda/yolo_preproc.cuh"
#include "vision/processors/cuda/fused_preproc.cuh"
#include "vision/processors/cuda/scrfd_preproc.cuh"
#include "vision/processors/cuda/draw_gpu.cuh"
#include <vector>

namespace modeldeploy::vision {
    // 惰性创建并返回持久 CUDA stream（backend 生命周期内复用）
    static cudaStream_t get_persistent_stream(void** slot) {
        cudaStream_t s = static_cast<cudaStream_t>(*slot);
        if (!s) {
            cudaStreamCreate(&s);
            *slot = s;
        }
        return s;
    }

    CudaProcessorBackend::~CudaProcessorBackend() {
        if (stream_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
            stream_ = nullptr;
        }
    }

    bool CudaProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                               const std::vector<int>& dst_size,
                                               float pad_val, LetterBoxRecord* record) {
        return yolo_preprocess_cuda(image, out, dst_size, pad_val, record,
                                    get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                                    const std::vector<int>& src_size,
                                                    int step_y, int step_uv, Tensor* out,
                                                    const std::vector<int>& dst_size,
                                                    float pad_val, LetterBoxRecord* record,
                                                    Device src_device) {
        if (src_device == Device::TPU) {
            MD_LOG_ERROR << "CudaProcessorBackend: NV12 src_device TPU not supported." << std::endl;
            return false;
        }
        // 内部用 cudaPointerGetAttributes 校验：CPU 走 H2D，GPU 内存零拷贝直接使用
        return yolo_preprocess_nv12_cuda(src_y, src_uv, src_size, step_y, step_uv,
                                         out, dst_size, pad_val, record,
                                         get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::scrfd_preprocess(const ImageData& image, Tensor* out,
                                                const std::vector<int>& dst_size,
                                                float pad_val, LetterBoxRecord* record) {
        return scrfd_preprocess_cuda(image, out, dst_size, pad_val, record,
                                     get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                      const std::vector<int>& dst_size,
                                                      float pad_val,
                                                      std::vector<LetterBoxRecord>* records) {
        return scrfd_preprocess_batch_cuda(images, out, dst_size, pad_val, records,
                                           &out_pool_, get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        return fused_preprocess_cuda(image.plane(0).data, {image.width(), image.height()},
                                     out, dst_size,
                                     origin_x, origin_y, scale_x, scale_y,
                                     alpha, beta, swap_rb, pad_value,
                                     get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                     const std::vector<int>& dst_size,
                                                     float pad_val,
                                                     std::vector<LetterBoxRecord>* records) {
        return yolo_preprocess_batch_cuda(images, out, dst_size, pad_val, records,
                                          get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::fused_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        return fused_preprocess_batch_cuda(images, out, dst_size, origins_x, origins_y,
                                           scales_x, scales_y, alpha, beta, swap_rb, pad_value,
                                           get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) {
        const float alpha[3] = {
            1.0f / 255.0f / std[0],
            1.0f / 255.0f / std[1],
            1.0f / 255.0f / std[2]
        };
        const float beta[3] = {
            -mean[0] / std[0],
            -mean[1] / std[1],
            -mean[2] / std[2]
        };
        const float pad[3] = {
            pad_value * alpha[0] + beta[0],
            pad_value * alpha[1] + beta[1],
            pad_value * alpha[2] + beta[2]
        };
        return fusion_rpnp_cuda(images, out, resize_sizes, dst_size,
                                std::vector<float>(alpha, alpha + 3),
                                std::vector<float>(beta, beta + 3), pad,
                                get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::draw_rect_nv12(ImageData& frame, float x, float y, float w, float h,
                                              float r, float g, float b, int thickness) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_rect_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, w, h,
                                  static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), thickness,
                                  get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_polygon_nv12(ImageData& frame, const std::vector<Point2f>& pts,
                                                 float r, float g, float b, int thickness) {
        std::vector<float> xs, ys;
        xs.reserve(pts.size()); ys.reserve(pts.size());
        for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_polygon_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                     frame.width(), frame.height(),
                                     pl0.step, pl1.step,
                                     xs.data(), ys.data(), static_cast<int>(xs.size()),
                                     static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                     static_cast<uint8_t>(b), thickness,
                                     get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_points_nv12(ImageData& frame, const std::vector<Point3f>& pts,
                                                float r, float g, float b, int radius) {
        std::vector<float> xs, ys;
        xs.reserve(pts.size()); ys.reserve(pts.size());
        for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_points_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                    frame.width(), frame.height(),
                                    pl0.step, pl1.step,
                                    xs.data(), ys.data(), static_cast<int>(xs.size()),
                                    static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                    static_cast<uint8_t>(b), radius,
                                    get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_text_nv12(ImageData& frame, float x, float y,
                                              const std::string& text,
                                              float r, float g, float b, int font_size) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_text_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, text.c_str(),
                                  static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), font_size,
                                  get_persistent_stream(&stream_));
    }
} // namespace modeldeploy::vision
