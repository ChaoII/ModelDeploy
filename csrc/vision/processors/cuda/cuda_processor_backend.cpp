//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#ifdef WITH_GPU
#include "vision/common/processors/yolo_preproc.cuh"
#include "vision/common/processors/fused_preproc.cuh"
#include "vision/face/face_det/scrfd_preproc.cuh"
#endif

namespace modeldeploy::vision {

bool CudaProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                           const std::vector<int>& dst_size,
                                           float pad_val, LetterBoxRecord* record) {
#ifdef WITH_GPU
    return yolo_preprocess_cuda(image, out, dst_size, pad_val, record);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::yolo_preprocess(image, out, dst_size, pad_val, record);
#endif
}

bool CudaProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                                const std::vector<int>& src_size,
                                                int step_y, int step_uv, Tensor* out,
                                                const std::vector<int>& dst_size,
                                                float pad_val, LetterBoxRecord* record) {
#ifdef WITH_GPU
    return yolo_preprocess_nv12_cuda(src_y, src_uv, src_size, step_y, step_uv,
                                     out, dst_size, pad_val, record);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::yolo_preprocess_nv12(src_y, src_uv, src_size, step_y, step_uv,
                                                     out, dst_size, pad_val, record);
#endif
}

bool CudaProcessorBackend::scrfd_preprocess(const ImageData& image, Tensor* out,
                                            const std::vector<int>& dst_size,
                                            float pad_val, LetterBoxRecord* record) {
#ifdef WITH_GPU
    return scrfd_preprocess_cuda(image, out, dst_size, pad_val, record);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::scrfd_preprocess(image, out, dst_size, pad_val, record);
#endif
}

bool CudaProcessorBackend::fused_preprocess(
    const ImageData& image, Tensor* out,
    const std::vector<int>& dst_size,
    float origin_x, float origin_y,
    float scale_x, float scale_y,
    const std::vector<float>& alpha,
    const std::vector<float>& beta,
    bool swap_rb, float pad_value) {
#ifdef WITH_GPU
    return fused_preprocess_cuda(image.data(), {image.width(), image.height()},
                                 out, dst_size,
                                 origin_x, origin_y, scale_x, scale_y,
                                 alpha, beta, swap_rb, pad_value);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::fused_preprocess(
        image, out, dst_size, origin_x, origin_y, scale_x, scale_y,
        alpha, beta, swap_rb, pad_value);
#endif
}

bool CudaProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                 const std::vector<int>& dst_size,
                                                 float pad_val,
                                                 std::vector<LetterBoxRecord>* records) {
#ifdef WITH_GPU
    return yolo_preprocess_batch_cuda(images, out, dst_size, pad_val, records);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::yolo_preprocess_batch(images, out, dst_size, pad_val, records);
#endif
}

bool CudaProcessorBackend::fused_preprocess_batch(
    const std::vector<ImageData>& images, Tensor* out,
    const std::vector<int>& dst_size,
    const std::vector<float>& origins_x, const std::vector<float>& origins_y,
    const std::vector<float>& scales_x, const std::vector<float>& scales_y,
    const std::vector<float>& alpha, const std::vector<float>& beta,
    bool swap_rb, float pad_value) {
#ifdef WITH_GPU
    return fused_preprocess_batch_cuda(images, out, dst_size, origins_x, origins_y,
                                       scales_x, scales_y, alpha, beta, swap_rb, pad_value);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::fused_preprocess_batch(
        images, out, dst_size, origins_x, origins_y, scales_x, scales_y,
        alpha, beta, swap_rb, pad_value);
#endif
}

bool CudaProcessorBackend::fusion_resize_pad_normalize_permute(
    const std::vector<ImageData>& images, Tensor* out,
    const std::vector<std::array<int, 2>>& resize_sizes,
    const std::vector<int>& dst_size,
    const std::vector<float>& mean, const std::vector<float>& std,
    float pad_value) {
#ifdef WITH_GPU
    const float alpha[3] = {1.0f / 255.0f / std[0],
                            1.0f / 255.0f / std[1],
                            1.0f / 255.0f / std[2]};
    const float beta[3] = {-mean[0] / std[0],
                           -mean[1] / std[1],
                           -mean[2] / std[2]};
    const float pad[3] = {pad_value * alpha[0] + beta[0],
                          pad_value * alpha[1] + beta[1],
                          pad_value * alpha[2] + beta[2]};
    return fusion_rpnp_cuda(images, out, resize_sizes, dst_size,
                            std::vector<float>(alpha, alpha + 3),
                            std::vector<float>(beta, beta + 3), pad);
#else
    MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, fallback to cpu" << std::endl;
    return CpuProcessorBackend::fusion_resize_pad_normalize_permute(
        images, out, resize_sizes, dst_size, mean, std, pad_value);
#endif
}

} // namespace modeldeploy::vision
