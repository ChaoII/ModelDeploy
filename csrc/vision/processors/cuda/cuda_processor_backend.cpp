//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#ifdef WITH_GPU
#include "vision/common/processors/yolo_preproc.cuh"
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

} // namespace modeldeploy::vision
