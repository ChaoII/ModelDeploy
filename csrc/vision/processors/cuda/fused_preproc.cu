//
// Created by aichao on 2025/8/2.
// 通用融合预处理 CUDA kernel，范式对齐 yolo_preproc.cu：thread_local workspace 池 + 单次 launch。
//

#include "vision/processors/cuda/fused_preproc.cuh"

// 源图像上传 workspace（线程局部，容量不足时扩容，避免每次调用 cudaMalloc）
struct FusedPreprocWorkspace {
    uint8_t* d_src = nullptr;
    size_t capacity = 0;

    ~FusedPreprocWorkspace() {
        if (d_src) cudaFree(d_src);
    }
};

static thread_local FusedPreprocWorkspace fused_ws;

// batch/rpnp 参数数组池：一次 cudaMalloc 打包全部 kernel 参数，跨调用复用
struct BatchParamWorkspace {
    uint8_t* d_ptr = nullptr;
    size_t capacity = 0;

    ~BatchParamWorkspace() {
        if (d_ptr) cudaFree(d_ptr);
    }
};
static thread_local BatchParamWorkspace param_ws;

namespace modeldeploy::vision {

    // 从后端缓冲池获取输出缓冲并零拷贝包装为 GPU Tensor（Tensor 不拥有设备内存）。
    static float* wrap_output_tensor(Tensor* output, CudaOutputBufferPool* pool,
                                     const std::vector<int64_t>& shape, DataType dtype,
                                     const std::string& name) {
        if (!pool) return nullptr;
        size_t numel = 1;
        for (int64_t d : shape) numel *= static_cast<size_t>(d);
        const size_t bytes = numel * Tensor::get_element_size(dtype);
        float* dst = pool->acquire(bytes);
        if (!dst) return nullptr;
        output->from_external_memory(dst, shape, dtype, [](void*) {}, Device::GPU, name);
        return dst;
    }

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
                           cudaStream_t stream,
                           CudaOutputBufferPool* dst_pool) {
    if (!out || src_size.size() != 2 || dst_size.size() != 2 ||
        alpha.size() != 3 || beta.size() != 3) {
        return false;
    }
    const int src_w = src_size[0];
    const int src_h = src_size[1];
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];

    // 1 output: GPU, FP32, CHW（从缓冲池获取，零拷贝包装）
    float* dst_ptr = wrap_output_tensor(out, dst_pool, {3, dst_h, dst_w},
                                        DataType::FP32, out->get_name());
    if (!dst_ptr) return false;

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
        dst_ptr,
        dst_h, dst_w,
        origin_x, origin_y, scale_x, scale_y,
        alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
        swap_rb, pad_value);
    const cudaError_t err = cudaGetLastError();
    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) return false;

    // 5 添加 batch 维
    out->expand_dim(0);
    return true;
}

__global__ void kernel_fused_preproc_batch(
    const uint8_t* __restrict__ src,
    const int* __restrict__ src_ws,
    const int* __restrict__ src_hs,
    const size_t* __restrict__ src_offsets,
    const float* __restrict__ origins_x,
    const float* __restrict__ origins_y,
    const float* __restrict__ scales_x,
    const float* __restrict__ scales_y,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float alpha0, const float beta0,
    const float alpha1, const float beta1,
    const float alpha2, const float beta2,
    const bool swap_rb,
    const float pad_value) {
    const int b = blockIdx.z;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int src_w = src_ws[b];
    const int src_h = src_hs[b];
    const float ox = origins_x[b];
    const float oy = origins_y[b];
    const float sx = scales_x[b];
    const float sy = scales_y[b];
    const uint8_t* src_b = src + src_offsets[b];
    float* dst_b = dst + static_cast<size_t>(b) * 3 * dst_h * dst_w;
    const int plane = dst_h * dst_w;
    const int dst_idx = y * dst_w + x;

    // 反推源坐标（最近邻）；浮点越界判断，避免负小数截断漏判
    const float src_xf = (static_cast<float>(x) - ox) / sx;
    const float src_yf = (static_cast<float>(y) - oy) / sy;
    if (src_xf < 0.0f || src_xf >= static_cast<float>(src_w) ||
        src_yf < 0.0f || src_yf >= static_cast<float>(src_h)) {
        dst_b[0 * plane + dst_idx] = pad_value;
        dst_b[1 * plane + dst_idx] = pad_value;
        dst_b[2 * plane + dst_idx] = pad_value;
        return;
    }
    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);
    const int idx = (src_y * src_w + src_x) * 3;
    const float bb = src_b[idx + 0];
    const float g = src_b[idx + 1];
    const float r = src_b[idx + 2];
    float v0, v1, v2;
    if (swap_rb) {
        v0 = r;
        v1 = g;
        v2 = bb;
    }
    else {
        v0 = bb;
        v1 = g;
        v2 = r;
    }
    dst_b[0 * plane + dst_idx] = v0 * alpha0 + beta0;
    dst_b[1 * plane + dst_idx] = v1 * alpha1 + beta1;
    dst_b[2 * plane + dst_idx] = v2 * alpha2 + beta2;
}

bool fused_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                 Tensor* out,
                                 const std::vector<int>& dst_size,
                                 const std::vector<float>& origins_x,
                                 const std::vector<float>& origins_y,
                                 const std::vector<float>& scales_x,
                                 const std::vector<float>& scales_y,
                                 const std::vector<float>& alpha,
                                 const std::vector<float>& beta,
                                 bool swap_rb, float pad_value,
                                 cudaStream_t stream,
                                 CudaOutputBufferPool* dst_pool) {
    if (images.empty() || dst_size.size() != 2) return false;
    const int batch = static_cast<int>(images.size());
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];
    // 整批输出：从缓冲池获取（复用），零拷贝包装
    float* dst_ptr = wrap_output_tensor(out, dst_pool, {batch, 3, dst_h, dst_w},
                                        DataType::FP32, out->get_name());
    if (!dst_ptr) return false;

    bool is_internal_stream = false;
    if (stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        is_internal_stream = true;
    }

    // 源图拼接上传 + 参数数组（batch 级一次性，避免 N 次 launch）
    std::vector<size_t> offsets(batch);
    std::vector<int> ws(batch), hs(batch);
    size_t total = 0;
    for (int b = 0; b < batch; ++b) {
        offsets[b] = total;
        ws[b] = images[b].width();
        hs[b] = images[b].height();
        total += static_cast<size_t>(hs[b]) * ws[b] * 3;
    }
    uint8_t* d_src = nullptr;
    int* d_ws = nullptr;
    int* d_hs = nullptr;
    size_t* d_offsets = nullptr;
    float* d_ox = nullptr;
    float* d_oy = nullptr;
    float* d_sx = nullptr;
    float* d_sy = nullptr;
    // 出错统一清理并返回失败（避免 goto）
    auto fail = [&]() -> bool {
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        cudaFree(d_src);
        return false;
    };
    if (cudaMalloc(&d_src, total) != cudaSuccess) return fail();
    for (int b = 0; b < batch; ++b) {
        if (cudaMemcpyAsync(d_src + offsets[b], images[b].plane(0).data,
                            static_cast<size_t>(hs[b]) * ws[b] * 3,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    }
    // 参数数组单块打包 + 线程局部池复用（避免每帧多次 cudaMalloc/cudaFree）
    const size_t need = sizeof(int) * batch * 2 + sizeof(size_t) * batch + sizeof(float) * batch * 4;
    if (param_ws.capacity < need) {
        if (param_ws.d_ptr) cudaFree(param_ws.d_ptr);
        cudaMalloc(&param_ws.d_ptr, need);
        param_ws.capacity = need;
    }
    {
        uint8_t* pbase = param_ws.d_ptr;
        d_ws = reinterpret_cast<int*>(pbase);
        d_hs = reinterpret_cast<int*>(pbase + sizeof(int) * batch);
        d_offsets = reinterpret_cast<size_t*>(pbase + sizeof(int) * batch * 2);
        float* pox = reinterpret_cast<float*>(pbase + sizeof(int) * batch * 2 + sizeof(size_t) * batch);
        float* poy = pox + batch;
        float* psx = poy + batch;
        float* psy = psx + batch;
        d_ox = pox; d_oy = poy; d_sx = psx; d_sy = psy;
    }
    if (cudaMemcpyAsync(d_ws, ws.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_hs, hs.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_offsets, offsets.data(), sizeof(size_t) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_ox, origins_x.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_oy, origins_y.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_sx, scales_x.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_sy, scales_y.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();

    {
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch);
        kernel_fused_preproc_batch<<<grid, block, 0, stream>>>(
            d_src, d_ws, d_hs, d_offsets, d_ox, d_oy, d_sx, d_sy,
            dst_ptr, dst_h, dst_w,
            alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
            swap_rb, pad_value);
        if (cudaGetLastError() != cudaSuccess) return fail();
    }

    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    cudaFree(d_src);
    return true;
}

// ═══════════ NV12 双平面融合预处理（crop/resize -> YUV2BGR -> normalize -> CHW，一次 launch）═══
__device__ inline void nv12_sample(
    const uint8_t* __restrict__ py, const uint8_t* __restrict__ puv,
    int src_w, int step_y, int step_uv, int sx, int sy,
    float& b, float& g, float& r) {
    const float yy = static_cast<float>(py[static_cast<size_t>(sy) * step_y + sx]);
    int uv_x = sx >> 1;
    int uv_y = sy >> 1;
    const uint8_t* uvp = puv + static_cast<size_t>(uv_y) * step_uv + (uv_x << 1);
    const float u = static_cast<float>(uvp[0]) - 128.0f;
    const float v = static_cast<float>(uvp[1]) - 128.0f;
    r = fminf(fmaxf(yy + 1.402f * v, 0.0f), 255.0f);
    g = fminf(fmaxf(yy - 0.344136f * u - 0.714136f * v, 0.0f), 255.0f);
    b = fminf(fmaxf(yy + 1.772f * u, 0.0f), 255.0f);
}

__global__ void kernel_fused_preproc_nv12(
    const uint8_t* __restrict__ src_y, const uint8_t* __restrict__ src_uv,
    const int src_h, const int src_w, const int step_y, const int step_uv,
    float* __restrict__ dst, const int dst_h, const int dst_w,
    const float origin_x, const float origin_y,
    const float scale_x, const float scale_y,
    const float alpha0, const float beta0,
    const float alpha1, const float beta1,
    const float alpha2, const float beta2,
    const bool swap_rb, const float pad_value) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<size_t>(dst_w) || y >= static_cast<size_t>(dst_h)) return;
    const float src_xf = (static_cast<float>(x) - origin_x) / scale_x;
    const float src_yf = (static_cast<float>(y) - origin_y) / scale_y;
    const int dst_idx = static_cast<int>(y) * dst_w + static_cast<int>(x);
    const int plane = dst_h * dst_w;
    if (src_xf < 0.0f || src_xf >= static_cast<float>(src_w) ||
        src_yf < 0.0f || src_yf >= static_cast<float>(src_h)) {
        dst[0 * plane + dst_idx] = pad_value;
        dst[1 * plane + dst_idx] = pad_value;
        dst[2 * plane + dst_idx] = pad_value;
        return;
    }
    const int src_x = static_cast<int>(src_xf);
    const int sy_i = static_cast<int>(src_yf);
    float b, g, r;
    nv12_sample(src_y, src_uv, src_w, step_y, step_uv, src_x, sy_i, b, g, r);
    float v0, v1, v2;
    if (swap_rb) { v0 = r; v1 = g; v2 = b; } else { v0 = b; v1 = g; v2 = r; }
    dst[0 * plane + dst_idx] = v0 * alpha0 + beta0;
    dst[1 * plane + dst_idx] = v1 * alpha1 + beta1;
    dst[2 * plane + dst_idx] = v2 * alpha2 + beta2;
}

__global__ void kernel_fused_preproc_nv12_batch(
    const uint8_t* __restrict__ const* src_ys,
    const uint8_t* __restrict__ const* src_uvs,
    const int* __restrict__ src_ws, const int* __restrict__ src_hs,
    const int* __restrict__ step_ys, const int* __restrict__ step_uvs,
    const float* __restrict__ origins_x, const float* __restrict__ origins_y,
    const float* __restrict__ scales_x, const float* __restrict__ scales_y,
    float* __restrict__ dst, const int dst_h, const int dst_w,
    const float alpha0, const float beta0,
    const float alpha1, const float beta1,
    const float alpha2, const float beta2,
    const bool swap_rb, const float pad_value) {
    const int b = blockIdx.z;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<size_t>(dst_w) || y >= static_cast<size_t>(dst_h)) return;
    const uint8_t* py = src_ys[b];
    const uint8_t* puv = src_uvs[b];
    const int src_w = src_ws[b];
    const int src_h = src_hs[b];
    const int step_y = step_ys[b];
    const int step_uv = step_uvs[b];
    const float sx = scales_x[b];
    const float sy = scales_y[b];
    const float src_xf = (static_cast<float>(x) - origins_x[b]) / sx;
    const float src_yf = (static_cast<float>(y) - origins_y[b]) / sy;
    float* dst_b = dst + static_cast<size_t>(b) * 3 * dst_h * dst_w;
    const int plane = dst_h * dst_w;
    const int dst_idx = static_cast<int>(y) * dst_w + static_cast<int>(x);
    if (src_xf < 0.0f || src_xf >= static_cast<float>(src_w) ||
        src_yf < 0.0f || src_yf >= static_cast<float>(src_h)) {
        dst_b[0 * plane + dst_idx] = pad_value;
        dst_b[1 * plane + dst_idx] = pad_value;
        dst_b[2 * plane + dst_idx] = pad_value;
        return;
    }
    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);
    float b_, g_, r_;
    nv12_sample(py, puv, src_w, step_y, step_uv, src_x, src_y, b_, g_, r_);
    float v0, v1, v2;
    if (swap_rb) { v0 = r_; v1 = g_; v2 = b_; } else { v0 = b_; v1 = g_; v2 = r_; }
    dst_b[0 * plane + dst_idx] = v0 * alpha0 + beta0;
    dst_b[1 * plane + dst_idx] = v1 * alpha1 + beta1;
    dst_b[2 * plane + dst_idx] = v2 * alpha2 + beta2;
}

// NV12 批参数 + host 缓冲池（thread_local 复用，避免每帧多次 cudaMalloc/free）
struct Nv12BatchWorkspace {
    uint8_t* d_host = nullptr;   // host NV12 帧打包缓冲（Y 后紧跟 UV）
    size_t host_cap = 0;
    uint8_t* d_arrays = nullptr; // 指针/尺寸/步长/坐标 数组打包
    size_t arrays_cap = 0;
    ~Nv12BatchWorkspace() {
        if (d_host) cudaFree(d_host);
        if (d_arrays) cudaFree(d_arrays);
    }
};
static thread_local Nv12BatchWorkspace nv12_ws;

bool fused_preprocess_nv12_cuda(const uint8_t* src_y, const uint8_t* src_uv,
                                const std::vector<int>& src_size,
                                int step_y, int step_uv,
                                Tensor* out,
                                const std::vector<int>& dst_size,
                                float origin_x, float origin_y,
                                float scale_x, float scale_y,
                                const std::vector<float>& alpha,
                                const std::vector<float>& beta,
                                bool swap_rb, float pad_value,
                                cudaStream_t stream,
                                CudaOutputBufferPool* dst_pool) {
    if (!out || src_size.size() != 2 || dst_size.size() != 2 ||
        alpha.size() != 3 || beta.size() != 3 || !src_y || !src_uv) {
        return false;
    }
    const int src_w = src_size[0];
    const int src_h = src_size[1];
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];
    if (step_y <= 0) step_y = src_w;
    if (step_uv <= 0) step_uv = src_w;

    float* dst_ptr = wrap_output_tensor(out, dst_pool, {3, dst_h, dst_w},
                                        DataType::FP32, out->get_name());
    if (!dst_ptr) return false;

    bool is_internal_stream = false;
    if (stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        is_internal_stream = true;
    }

    cudaPointerAttributes attr{};
    const bool y_on_dev = cudaPointerGetAttributes(&attr, src_y) == cudaSuccess &&
                          attr.type == cudaMemoryTypeDevice;
    const uint8_t* d_y = src_y;
    const uint8_t* d_uv = src_uv;
    if (!y_on_dev) {
        const size_t y_bytes = static_cast<size_t>(src_h) * src_w;
        const size_t uv_bytes = static_cast<size_t>(src_h / 2) * src_w;
        if (fused_ws.capacity < y_bytes + uv_bytes) {
            if (fused_ws.d_src) cudaFree(fused_ws.d_src);
            cudaMalloc(&fused_ws.d_src, y_bytes + uv_bytes);
            fused_ws.capacity = y_bytes + uv_bytes;
        }
        cudaMemcpyAsync(fused_ws.d_src, src_y, y_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(fused_ws.d_src + y_bytes, src_uv, uv_bytes, cudaMemcpyHostToDevice, stream);
        d_y = fused_ws.d_src;
        d_uv = fused_ws.d_src + y_bytes;
    }

    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    kernel_fused_preproc_nv12<<<grid, block, 0, stream>>>(
        d_y, d_uv, src_h, src_w, step_y, step_uv, dst_ptr, dst_h, dst_w,
        origin_x, origin_y, scale_x, scale_y,
        alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
        swap_rb, pad_value);
    const cudaError_t err = cudaGetLastError();
    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    if (err != cudaSuccess) return false;

    out->expand_dim(0);
    return true;
}

bool fused_preprocess_nv12_batch_cuda(const std::vector<ImageData>& images,
                                      Tensor* out,
                                      const std::vector<int>& dst_size,
                                      const std::vector<float>& origins_x,
                                      const std::vector<float>& origins_y,
                                      const std::vector<float>& scales_x,
                                      const std::vector<float>& scales_y,
                                      const std::vector<float>& alpha,
                                      const std::vector<float>& beta,
                                      bool swap_rb, float pad_value,
                                      cudaStream_t stream,
                                      CudaOutputBufferPool* dst_pool) {
    if (images.empty() || dst_size.size() != 2) return false;
    const int batch = static_cast<int>(images.size());
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];
    float* dst_ptr = wrap_output_tensor(out, dst_pool, {batch, 3, dst_h, dst_w},
                                        DataType::FP32, out->get_name());
    if (!dst_ptr) return false;

    bool is_internal_stream = false;
    if (stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        is_internal_stream = true;
    }
    auto fail = [&]() -> bool {
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        return false;
    };

    // 1) 收集每帧设备侧 Y/UV 指针（host 帧打包上传到 pool）
    std::vector<const uint8_t*> y_list(batch), uv_list(batch);
    std::vector<int> ws(batch), hs(batch), sty(batch), stuv(batch);
    std::vector<size_t> offs(batch);
    size_t host_total = 0;
    for (int b = 0; b < batch; ++b) {
        const auto& im = images[b];
        ws[b] = im.width(); hs[b] = im.height();
        sty[b] = im.plane(0).step > 0 ? im.plane(0).step : ws[b];
        stuv[b] = im.plane_count() > 1 ? (im.plane(1).step > 0 ? im.plane(1).step : ws[b]) : ws[b];
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, im.plane(0).data) == cudaSuccess &&
            attr.type == cudaMemoryTypeDevice) {
            y_list[b] = im.plane(0).data;
            uv_list[b] = im.plane_count() > 1 ? im.plane(1).data : nullptr;
            offs[b] = SIZE_MAX; // marker: device, no host slot
        } else {
            offs[b] = host_total;
            host_total += static_cast<size_t>(hs[b]) * ws[b] + static_cast<size_t>(hs[b] / 2) * ws[b];
        }
    }
    if (host_total > 0) {
        if (nv12_ws.host_cap < host_total) {
            if (nv12_ws.d_host) cudaFree(nv12_ws.d_host);
            cudaMalloc(&nv12_ws.d_host, host_total);
            nv12_ws.host_cap = host_total;
        }
        for (int b = 0; b < batch; ++b) {
            if (offs[b] == SIZE_MAX) continue;
            const size_t yb = static_cast<size_t>(hs[b]) * ws[b];
            if (cudaMemcpyAsync(nv12_ws.d_host + offs[b], images[b].plane(0).data, yb,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
            if (cudaMemcpyAsync(nv12_ws.d_host + offs[b] + yb, images[b].plane(1).data,
                                static_cast<size_t>(hs[b] / 2) * ws[b],
                                cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
        }
        for (int b = 0; b < batch; ++b) {
            if (offs[b] == SIZE_MAX) continue;
            y_list[b] = nv12_ws.d_host + offs[b];
            uv_list[b] = y_list[b] + static_cast<size_t>(hs[b]) * ws[b];
        }
    }

    // 2) 参数数组打包（指针 + 尺寸 + 坐标）
    const size_t need = sizeof(uint8_t*) * batch * 2 + sizeof(int) * batch * 4 +
                        sizeof(float) * batch * 4;
    if (nv12_ws.arrays_cap < need) {
        if (nv12_ws.d_arrays) cudaFree(nv12_ws.d_arrays);
        cudaMalloc(&nv12_ws.d_arrays, need);
        nv12_ws.arrays_cap = need;
    }
    uint8_t* abase = nv12_ws.d_arrays;
    uint8_t** d_yptrs = reinterpret_cast<uint8_t**>(abase);
    uint8_t** d_uvptrs = reinterpret_cast<uint8_t**>(abase + sizeof(uint8_t*) * batch);
    int* d_ws = reinterpret_cast<int*>(abase + sizeof(uint8_t*) * batch * 2);
    int* d_hs = d_ws + batch;
    int* d_sty = d_hs + batch;
    int* d_stuv = d_sty + batch;
    float* d_ox = reinterpret_cast<float*>(d_stuv + batch);
    float* d_oy = d_ox + batch;
    float* d_sx = d_oy + batch;
    float* d_sy = d_sx + batch;
    std::vector<const uint8_t*> yptrs(y_list), uvptrs(uv_list);
    if (cudaMemcpyAsync(d_yptrs, yptrs.data(), sizeof(uint8_t*) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_uvptrs, uvptrs.data(), sizeof(uint8_t*) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_ws, ws.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_hs, hs.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_sty, sty.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_stuv, stuv.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_ox, origins_x.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_oy, origins_y.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_sx, scales_x.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_sy, scales_y.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();

    {
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch);
        kernel_fused_preproc_nv12_batch<<<grid, block, 0, stream>>>(
            d_yptrs, d_uvptrs, d_ws, d_hs, d_sty, d_stuv,
            d_ox, d_oy, d_sx, d_sy, dst_ptr, dst_h, dst_w,
            alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
            swap_rb, pad_value);
        if (cudaGetLastError() != cudaSuccess) return fail();
    }
    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    return true;
}

__global__ void kernel_fusion_rpnp_batch(
    const uint8_t* __restrict__ src,
    const int* __restrict__ src_ws,
    const int* __restrict__ src_hs,
    const size_t* __restrict__ src_offsets,
    const int* __restrict__ resize_ws,
    const int* __restrict__ resize_hs,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float alpha0, const float beta0,
    const float alpha1, const float beta1,
    const float alpha2, const float beta2,
    const float pad0, const float pad1, const float pad2) {
    const int b = blockIdx.z;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int src_w = src_ws[b];
    const int src_h = src_hs[b];
    const int resize_w = resize_ws[b];
    const int resize_h = resize_hs[b];
    const uint8_t* src_b = src + src_offsets[b];
    float* dst_b = dst + static_cast<size_t>(b) * 3 * dst_h * dst_w;
    const int plane = dst_h * dst_w;
    const int idx = y * dst_w + x;

    // pad（右 & 下），逐通道仿射后空间
    if (y >= resize_h || x >= resize_w) {
        dst_b[0 * plane + idx] = pad0;
        dst_b[1 * plane + idx] = pad1;
        dst_b[2 * plane + idx] = pad2;
        return;
    }
    // 预计算 dst->src 映射系数（乘替代除）
    const float kx = static_cast<float>(src_w) / resize_w;
    const float ky = static_cast<float>(src_h) / resize_h;
    const int sx_v = static_cast<int>(static_cast<float>(x) * kx);
    const int sy_v = static_cast<int>(static_cast<float>(y) * ky);
    const int sx = sx_v < src_w - 1 ? sx_v : (src_w - 1);
    const int sy = sy_v < src_h - 1 ? sy_v : (src_h - 1);
    const uint8_t* p = src_b + (sy * src_w + sx) * 3;
    const float bb = p[0];
    const float g = p[1];
    const float r = p[2];
    // swap BGR->RGB：C0=R, C1=G, C2=B
    dst_b[0 * plane + idx] = r * alpha0 + beta0;
    dst_b[1 * plane + idx] = g * alpha1 + beta1;
    dst_b[2 * plane + idx] = bb * alpha2 + beta2;
}

bool fusion_rpnp_cuda(const std::vector<ImageData>& images,
                      Tensor* out,
                      const std::vector<std::array<int, 2>>& resize_sizes,
                      const std::vector<int>& dst_size,
                      const std::vector<float>& alpha,
                      const std::vector<float>& beta,
                      const float pad[3],
                      cudaStream_t stream,
                      CudaOutputBufferPool* dst_pool) {
    if (images.empty() || dst_size.size() != 2) return false;
    const int batch = static_cast<int>(images.size());
    const int dst_w = dst_size[0];
    const int dst_h = dst_size[1];
    // 整批输出：从缓冲池获取（复用），零拷贝包装
    float* dst_ptr = wrap_output_tensor(out, dst_pool, {batch, 3, dst_h, dst_w},
                                        DataType::FP32, out->get_name());
    if (!dst_ptr) return false;

    bool is_internal_stream = false;
    if (stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        is_internal_stream = true;
    }

    std::vector<size_t> offsets(batch);
    std::vector<int> ws(batch), hs(batch), rws(batch), rhs(batch);
    size_t total = 0;
    for (int b = 0; b < batch; ++b) {
        offsets[b] = total;
        ws[b] = images[b].width();
        hs[b] = images[b].height();
        rws[b] = resize_sizes[b][0];
        rhs[b] = resize_sizes[b][1];
        total += static_cast<size_t>(hs[b]) * ws[b] * 3;
    }

    uint8_t* d_src = nullptr;
    int* d_ws = nullptr;
    int* d_hs = nullptr;
    size_t* d_offsets = nullptr;
    int* d_rws = nullptr;
    int* d_rhs = nullptr;
    // 出错统一清理并返回失败（避免 goto）
    auto fail = [&]() -> bool {
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        cudaFree(d_src);
        return false;
    };
    if (cudaMalloc(&d_src, total) != cudaSuccess) return fail();
    for (int b = 0; b < batch; ++b) {
        if (cudaMemcpyAsync(d_src + offsets[b], images[b].plane(0).data,
                            static_cast<size_t>(hs[b]) * ws[b] * 3,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    }
    // 参数数组单块打包 + 线程局部池复用
    const size_t need = sizeof(int) * batch * 4 + sizeof(size_t) * batch;
    if (param_ws.capacity < need) {
        if (param_ws.d_ptr) cudaFree(param_ws.d_ptr);
        cudaMalloc(&param_ws.d_ptr, need);
        param_ws.capacity = need;
    }
    {
        uint8_t* pbase = param_ws.d_ptr;
        d_ws = reinterpret_cast<int*>(pbase);
        d_hs = reinterpret_cast<int*>(pbase + sizeof(int) * batch);
        d_rws = reinterpret_cast<int*>(pbase + sizeof(int) * batch * 2);
        d_rhs = reinterpret_cast<int*>(pbase + sizeof(int) * batch * 3);
        d_offsets = reinterpret_cast<size_t*>(pbase + sizeof(int) * batch * 4);
    }
    if (cudaMemcpyAsync(d_ws, ws.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_hs, hs.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_rws, rws.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_rhs, rhs.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    if (cudaMemcpyAsync(d_offsets, offsets.data(), sizeof(size_t) * batch, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();

    {
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch);
        kernel_fusion_rpnp_batch<<<grid, block, 0, stream>>>(
            d_src, d_ws, d_hs, d_offsets, d_rws, d_rhs,
            dst_ptr, dst_h, dst_w,
            alpha[0], beta[0], alpha[1], beta[1], alpha[2], beta[2],
            pad[0], pad[1], pad[2]);
        if (cudaGetLastError() != cudaSuccess) return fail();
    }

    cudaStreamSynchronize(stream);
    if (is_internal_stream) cudaStreamDestroy(stream);
    cudaFree(d_src);
    return true;
}

} // namespace modeldeploy::vision
