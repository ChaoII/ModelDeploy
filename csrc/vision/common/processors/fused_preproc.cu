//
// Created by aichao on 2025/8/2.
// 通用融合预处理 CUDA kernel，范式对齐 yolo_preproc.cu：thread_local workspace 池 + 单次 launch。
//

#include "vision/common/processors/fused_preproc.cuh"

// 源图像上传 workspace（线程局部，容量不足时扩容，避免每次调用 cudaMalloc）
struct FusedPreprocWorkspace {
    uint8_t* d_src = nullptr;
    size_t capacity = 0;

    ~FusedPreprocWorkspace() {
        if (d_src) cudaFree(d_src);
    }
};

static thread_local FusedPreprocWorkspace fused_ws;

namespace modeldeploy::vision {

__global__ void kernel_fused_preproc(
    const uint8_t* __restrict__ src,
    const int src_h,
    const int src_w,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float origin_x,
    const float origin_y,
    const float scale_x,
    const float scale_y,
    const float alpha0, const float beta0,
    const float alpha1, const float beta1,
    const float alpha2, const float beta2,
    const bool swap_rb,
    const float pad_value) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // 反推源坐标（最近邻）；越界判断用浮点，避免负小数截断漏判
    const float src_xf = (static_cast<float>(x) - origin_x) / scale_x;
    const float src_yf = (static_cast<float>(y) - origin_y) / scale_y;

    const int dst_idx = y * dst_w + x;
    const int plane_size = dst_h * dst_w;

    if (src_xf < 0.0f || src_xf >= static_cast<float>(src_w) ||
        src_yf < 0.0f || src_yf >= static_cast<float>(src_h)) {
        // 越界填充区（letterbox / OCR 右/下 pad），pad_value 已是仿射后（归一化）空间
        dst[0 * plane_size + dst_idx] = pad_value;
        dst[1 * plane_size + dst_idx] = pad_value;
        dst[2 * plane_size + dst_idx] = pad_value;
        return;
    }

    // BGR 打包: [B0G0R0 B1G1R1 ...]
    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);
    const int src_idx = (src_y * src_w + src_x) * 3;
    const float b = src[src_idx + 0];
    const float g = src[src_idx + 1];
    const float r = src[src_idx + 2];
    float v0, v1, v2;
    if (swap_rb) {
        // BGR -> RGB：输出 C0=R, C1=G, C2=B
        v0 = r;
        v1 = g;
        v2 = b;
    }
    else {
        v0 = b;
        v1 = g;
        v2 = r;
    }

    // 仿射 + 写入 CHW（C0,R / C1,G / C2,B）
    dst[0 * plane_size + dst_idx] = v0 * alpha0 + beta0;
    dst[1 * plane_size + dst_idx] = v1 * alpha1 + beta1;
    dst[2 * plane_size + dst_idx] = v2 * alpha2 + beta2;
}

bool fused_preprocess_cuda(const uint8_t* src,
                           const std::vector<int>& src_size,
                           Tensor* out,
                           const std::vector<int>& dst_size,
                           float origin_x, float origin_y,
                           float scale_x, float scale_y,
                           const std::vector<float>& alpha,
                           const std::vector<float>& beta,
                           bool swap_rb,
                           float pad_value,
                           cudaStream_t stream) {
    if (!out || src_size.size() != 2 || dst_size.size() != 2 ||
        alpha.size() != 3 || beta.size() != 3) {
        return false;
    }
    const int src_w = src_size[0];
    const int src_h = src_size[1];
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];

    // 1 output: GPU, FP32, CHW
    out->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

    // 2 CUDA stream
    bool is_internal_stream = false;
    if (stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        is_internal_stream = true;
    }

    // 3 上传源图（若已是设备内存则零拷贝）
    const size_t src_bytes = static_cast<size_t>(src_h) * src_w * 3;
    const uint8_t* d_src = nullptr;
    cudaPointerAttributes attr{};
    const bool is_device =
        cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
    if (is_device) {
        d_src = src;
    }
    else {
        if (fused_ws.capacity < src_bytes) {
            if (fused_ws.d_src) cudaFree(fused_ws.d_src);
            cudaMalloc(&fused_ws.d_src, src_bytes);
            fused_ws.capacity = src_bytes;
        }
        cudaMemcpyAsync(fused_ws.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
        d_src = fused_ws.d_src;
    }

    // 4 launch kernel
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    kernel_fused_preproc<<<grid, block, 0, stream>>>(
        d_src, src_h, src_w,
        out->data_ptr<float>(),
        dst_h, dst_w,
        origin_x, origin_y, scale_x, scale_y,
        alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
        swap_rb, pad_value);
    const cudaError_t err = cudaGetLastError();
    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) return false;

    // 5 增加 batch 维
    out->expand_dim(0);
    return true;
}

} // namespace modeldeploy::vision
