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
            // NonBlocking：不与默认流(stream 0)隐式同步。否则与 20 路解码默认流上的
            // 同步 D2D 拷贝串行化，批推理耗时膨胀 ~5×（实测 23ms→4.6ms/batch-8）。
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
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

    bool CudaProcessorBackend::crop(const ImageData& image, float x, float y,
                                    float w, float h, ImageData* out) {
        if (!out || image.type() != MdImageType::NV12 || image.plane_count() < 2) return false;
        const int iw = image.width(), ih = image.height();
        int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
        int x1 = static_cast<int>(x + w), y1 = static_cast<int>(y + h);
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > iw) x1 = iw; if (y1 > ih) y1 = ih;
        if (x1 <= x0 || y1 <= y0) return false;
        x0 &= ~1; y0 &= ~1; x1 &= ~1; y1 &= ~1;   // UV 偶数对齐
        if (x1 <= x0 || y1 <= y0) return false;
        const int cw = x1 - x0, ch = y1 - y0;     // 偶数
        const auto py = image.plane(0);
        const auto puv = image.plane(1);
        const int step_y = py.step > 0 ? py.step : iw;
        const int step_uv = puv.step > 0 ? puv.step : iw;
        const size_t ybytes = static_cast<size_t>(cw) * ch;
        const size_t uvbytes = static_cast<size_t>(cw) * (ch / 2);  // 每行 cw 字节(交错)，ch/2 行
        uint8_t* dbuf = nullptr;
        if (cudaMalloc(&dbuf, ybytes + uvbytes) != cudaSuccess) return false;
        // 在持久 stream 上拷贝 + 同步，避免与后续消费该裁剪块的非阻塞 stream 产生跨流竞争
        // （同步默认流 D2D 拷贝在多次连续调用后可能与持久流 kernel 竞速）。
        cudaStream_t cstream = get_persistent_stream(&stream_);
        cudaMemcpy2DAsync(dbuf, static_cast<size_t>(cw),
                          py.data + static_cast<size_t>(y0) * step_y + x0,
                          static_cast<size_t>(step_y),
                          static_cast<size_t>(cw), static_cast<size_t>(ch),
                          cudaMemcpyDeviceToDevice, cstream);
        cudaMemcpy2DAsync(dbuf + ybytes, static_cast<size_t>(cw),
                          puv.data + static_cast<size_t>(y0 / 2) * step_uv + x0,
                          static_cast<size_t>(step_uv),
                          static_cast<size_t>(cw), static_cast<size_t>(ch / 2),
                          cudaMemcpyDeviceToDevice, cstream);
        cudaStreamSynchronize(cstream);
        std::shared_ptr<void> owner(dbuf, [](void* p) { if (p) cudaFree(p); });
        ImageData::Plane pl[2] = {{dbuf, cw}, {dbuf + ybytes, cw}};
        *out = ImageData::from_planes(pl, 2, MdImageType::NV12, cw, ch,
                                      Device::GPU, std::move(owner));
        return !out->empty();
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

    bool CudaProcessorBackend::fused_preprocess_common(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        // NV12 双平面源 → NV12 融合 kernel（一次 launch：crop/resize + YUV2BGR + norm→CHW）
        if (image.type() == MdImageType::NV12 && image.plane_count() >= 2) {
            return fused_preprocess_common_nv12_cuda(image.plane(0).data, image.plane(1).data,
                                              {image.width(), image.height()},
                                              image.plane(0).step, image.plane(1).step,
                                              out, dst_size,
                                              origin_x, origin_y, scale_x, scale_y,
                                              alpha, beta, swap_rb, pad_value,
                                              get_persistent_stream(&stream_), &out_pool_);
        }
        return fused_preprocess_common_cuda(image.plane(0).data, {image.width(), image.height()},
                                     out, dst_size,
                                     origin_x, origin_y, scale_x, scale_y,
                                     alpha, beta, swap_rb, pad_value,
                                     get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                     const std::vector<int>& dst_size,
                                                     float pad_val,
                                                     std::vector<LetterBoxRecord>* records) {
        // NV12 帧（设备/host）→ NV12 融合 kernel：设备帧零 PCIe，host 帧聚合 H2D
        if (!images.empty()) {
            bool all_nv12 = true;
            for (const auto& im : images) {
                if (im.type() != MdImageType::NV12 || im.plane_count() < 2) {
                    all_nv12 = false;
                    break;
                }
            }
            if (all_nv12) {
                return yolo_preprocess_nv12_batch_cuda(images, out, dst_size, pad_val, records,
                                                       get_persistent_stream(&stream_), &out_pool_);
            }
        }
        return yolo_preprocess_batch_cuda(images, out, dst_size, pad_val, records,
                                          get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::fused_preprocess_common_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        // 全 NV12 批（device/host 混合亦可）→ 单次 NV12 融合 kernel
        if (!images.empty()) {
            bool all_nv12 = true;
            for (const auto& im : images) {
                if (im.type() != MdImageType::NV12 || im.plane_count() < 2) {
                    all_nv12 = false;
                    break;
                }
            }
            if (all_nv12) {
                return fused_preprocess_common_nv12_batch_cuda(images, out, dst_size,
                                                        origins_x, origins_y,
                                                        scales_x, scales_y,
                                                        alpha, beta, swap_rb, pad_value,
                                                        get_persistent_stream(&stream_), &out_pool_);
            }
        }
        return fused_preprocess_common_batch_cuda(images, out, dst_size, origins_x, origins_y,
                                           scales_x, scales_y, alpha, beta, swap_rb, pad_value,
                                           get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::ocr_det_preprocess(
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
        return ocr_det_preprocess_cuda(images, out, resize_sizes, dst_size,
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

    // ── 设备侧高层可视化（暂为空实现，后续任务替换为真实实现）──
    bool CudaProcessorBackend::vis_det_nv12(ImageData& frame,
                                            const std::vector<DetectionResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_obb_nv12(ImageData& frame,
                                            const std::vector<ObbResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_pose_nv12(ImageData& frame,
                                             const std::vector<KeyPointsResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_keypoints_nv12(ImageData& frame,
                                                  const std::vector<KeyPointsResult>& result,
                                                  const VisionProcessorBackend::VisOptions& opt,
                                                  bool draw_lines) {
        (void)frame; (void)result; (void)opt; (void)draw_lines; return false;
    }
    bool CudaProcessorBackend::vis_hand_nv12(ImageData& frame,
                                             const std::vector<KeyPointsResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_ocr_nv12(ImageData& frame, const OCRResult& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_lpr_nv12(ImageData& frame,
                                            const std::vector<LprResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_attr_nv12(ImageData& frame,
                                             const std::vector<AttributeResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt,
                                             const std::vector<int>& abnormal_ids,
                                             bool show_attr) {
        (void)frame; (void)result; (void)opt; (void)abnormal_ids; (void)show_attr; return false;
    }
    bool CudaProcessorBackend::vis_cls_nv12(ImageData& frame, const ClassifyResult& result,
                                            const VisionProcessorBackend::VisOptions& opt,
                                            int top_k) {
        (void)frame; (void)result; (void)opt; (void)top_k; return false;
    }
    bool CudaProcessorBackend::vis_iseg_nv12(ImageData& frame,
                                             const std::vector<InstanceSegResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_sem_nv12(ImageData& frame, const SemSegResult& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        (void)frame; (void)result; (void)opt; return false;
    }
    bool CudaProcessorBackend::vis_depth_nv12(ImageData& frame, const DepthResult& result,
                                              const VisionProcessorBackend::VisOptions& opt,
                                              bool colorize) {
        (void)frame; (void)result; (void)opt; (void)colorize; return false;
    }
} // namespace modeldeploy::vision
