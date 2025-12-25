#include "vision/common/processors/yolo_preproc.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>


__constant__ float kMean[3] = {0.f, 0.f, 0.f};
__constant__ float kStd[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

__global__ void kernel_yolo_preproc(
    const uint8_t* __restrict__ src,
    const int src_h,
    const int src_w,
    const int src_step,
    float* __restrict__ dst,
    const int dst_h, const int dst_w,
    const float scale,
    const float pad_w,
    const float pad_h,
    const float pad_value) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const float src_xf = (x - pad_w) / scale;
    const float src_yf = (y - pad_h) / scale;

    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    const int dst_idx = y * dst_w + x;
    const int plane = dst_h * dst_w;

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        const float v0 = (pad_value - kMean[0]) * kStd[0];
        const float v1 = (pad_value - kMean[1]) * kStd[1];
        const float v2 = (pad_value - kMean[2]) * kStd[2];
        dst[dst_idx] = v0;
        dst[plane + dst_idx] = v1;
        dst[2 * plane + dst_idx] = v2;
    }
    else {
        const int src_idx = src_y * src_step + src_x * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        dst[dst_idx] = (r - kMean[0]) * kStd[0];
        dst[plane + dst_idx] = (g - kMean[1]) * kStd[1];
        dst[2 * plane + dst_idx] = (b - kMean[2]) * kStd[2];
    }
}

struct PreprocWorkspace {
    uint8_t* d_src = nullptr;
    size_t capacity = 0;

    ~PreprocWorkspace() {
        if (d_src) {
            cudaFree(d_src);
        }
    }
};

static thread_local PreprocWorkspace ws;

inline void compute_letterbox(
    const int src_h,
    const int src_w,
    const int dst_h,
    const int dst_w,
    float& scale,
    float& pad_w,
    float& pad_h) {
    scale = std::min(
        static_cast<float>(dst_h) / src_h,
        static_cast<float>(dst_w) / src_w);
    const float new_h = src_h * scale;
    const float new_w = src_w * scale;
    pad_h = (dst_h - new_h) * 0.5f;
    pad_w = (dst_w - new_w) * 0.5f;
}

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(
        ImageData* image,
        Tensor* output,
        const std::vector<int>& dst_size,
        const std::vector<float>& pad_val,
        LetterBoxRecord* letter_box_record) {
        if (!image || !output || dst_size.size() != 2 || pad_val.empty()) {
            return false;
        }
        const int src_h = image->height();
        const int src_w = image->width();
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        const float pad_value = pad_val[0];

        // 1️⃣ output: GPU, FP32, CHW
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

        // 2️⃣ CUDA stream（保持你原来的行为）
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return false;
        }

        // 3️⃣ letterbox（host 计算）
        float scale, pad_w, pad_h;
        compute_letterbox(src_h, src_w, dst_h, dst_w, scale, pad_w, pad_h);

        if (letter_box_record) {
            letter_box_record->ipt_w = static_cast<float>(src_w);
            letter_box_record->ipt_h = static_cast<float>(src_h);
            letter_box_record->out_w = static_cast<float>(dst_w);
            letter_box_record->out_h = static_cast<float>(dst_h);
            letter_box_record->scale = scale;
            letter_box_record->pad_w = pad_w;
            letter_box_record->pad_h = pad_h;
        }

        // 4️⃣ src 指针处理（workspace，避免反复 malloc）
        const uint8_t* src = image->data();
        const size_t src_bytes = static_cast<size_t>(src_h) * src_w * image->channels();
        const uint8_t* d_src = nullptr;
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        if (is_device) {
            d_src = src;
        }
        else {
            if (ws.capacity < src_bytes) {
                if (ws.d_src) cudaFree(ws.d_src);
                cudaMalloc(&ws.d_src, src_bytes);
                ws.capacity = src_bytes;
            }
            cudaMemcpyAsync(ws.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
            d_src = ws.d_src;
        }
        // 5️⃣ launch kernel
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        kernel_yolo_preproc<<<grid, block, 0, stream>>>(
            d_src,
            src_h,
            src_w,
            src_w * 3,
            output->data_ptr<float>(),
            dst_h,
            dst_w,
            scale,
            pad_w,
            pad_h,
            pad_value);
        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            return false;
        }
        // 6️⃣ batch 维（保持你原行为）
        output->expand_dim(0);
        return true;
    }
} // namespace modeldeploy::vision
