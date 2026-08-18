#include "vision/processors/cuda/scrfd_preproc.cuh"
#include <cuda_runtime.h>
#include <vision/utils.h>

// SCRFD: alpha = 1/128, beta = -127.5/128 ≈ -0.996094, pad = 0.0
__constant__ float scrfd_alpha[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
__constant__ float scrfd_beta[3] = {-127.5f / 128.f, -127.5f / 128.f, -127.5f / 128.f};

__global__ void scrfd_kernel_bgr_fusion(
    const uint8_t* __restrict__ src,
    const int src_h,
    const int src_w,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float scale,
    const float pad_w,
    const float pad_h,
    const float pad_value) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    const float src_xf = (x - pad_w) / scale;
    const float src_yf = (y - pad_h) / scale;

    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    const int dst_idx = y * dst_w + x;
    const int plane_size = dst_h * dst_w;

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        dst[0 * plane_size + dst_idx] = pad_value * scrfd_alpha[0] + scrfd_beta[0];
        dst[1 * plane_size + dst_idx] = pad_value * scrfd_alpha[1] + scrfd_beta[1];
        dst[2 * plane_size + dst_idx] = pad_value * scrfd_alpha[2] + scrfd_beta[2];
    }
    else {
        const int src_idx = (src_y * src_w + src_x) * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        dst[0 * plane_size + dst_idx] = r * scrfd_alpha[0] + scrfd_beta[0];
        dst[1 * plane_size + dst_idx] = g * scrfd_alpha[1] + scrfd_beta[1];
        dst[2 * plane_size + dst_idx] = b * scrfd_alpha[2] + scrfd_beta[2];
    }
}

__global__ void scrfd_kernel_nv12_fusion(
    const uint8_t* __restrict__ srcY,
    const uint8_t* __restrict__ srcUV,
    const int src_h,
    const int src_w,
    const int stepY,
    const int stepUV,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float scale,
    const float pad_w,
    const float pad_h,
    const float pad_value) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    const float src_xf = (x - pad_w) / scale;
    const float src_yf = (y - pad_h) / scale;

    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    const int dst_idx = y * dst_w + x;
    const int plane_size = dst_h * dst_w;

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        dst[0 * plane_size + dst_idx] = pad_value * scrfd_alpha[0] + scrfd_beta[0];
        dst[1 * plane_size + dst_idx] = pad_value * scrfd_alpha[1] + scrfd_beta[1];
        dst[2 * plane_size + dst_idx] = pad_value * scrfd_alpha[2] + scrfd_beta[2];
    }
    else {
        const float y_val = srcY[src_y * stepY + src_x];

        const int uv_x = src_x >> 1;
        const int uv_y = src_y >> 1;

        const int safe_uv_x = min(uv_x, (src_w >> 1) - 1);
        const int safe_uv_y = min(uv_y, (src_h >> 1) - 1);

        const uint8_t* uv_row = srcUV + safe_uv_y * stepUV;
        float u_val = uv_row[safe_uv_x * 2 + 0];
        float v_val = uv_row[safe_uv_x * 2 + 1];

        u_val -= 128.0f;
        v_val -= 128.0f;

        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
        float b = y_val + 1.772f * u_val;

        r = fminf(fmaxf(r, 0.0f), 255.0f);
        g = fminf(fmaxf(g, 0.0f), 255.0f);
        b = fminf(fmaxf(b, 0.0f), 255.0f);

        dst[0 * plane_size + dst_idx] = r * scrfd_alpha[0] + scrfd_beta[0];
        dst[1 * plane_size + dst_idx] = g * scrfd_alpha[1] + scrfd_beta[1];
        dst[2 * plane_size + dst_idx] = b * scrfd_alpha[2] + scrfd_beta[2];
    }
}

// SCRFD 整批融合 kernel：blockIdx.z = batch 内索引，每图独立 letterbox 映射 + 归一化 (x-127.5)/128
__global__ void scrfd_kernel_bgr_fusion_batch(
    const uint8_t* __restrict__ src,
    const int* __restrict__ src_ws,
    const int* __restrict__ src_hs,
    const size_t* __restrict__ src_offsets,
    const float* __restrict__ scales,
    const float* __restrict__ pad_ws,
    const float* __restrict__ pad_hs,
    float* __restrict__ dst,
    const int dst_h,
    const int dst_w,
    const float pad_value) {
    const int b = blockIdx.z;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int src_w = src_ws[b];
    const int src_h = src_hs[b];
    const float scale = scales[b];
    const float pad_w = pad_ws[b];
    const float pad_h = pad_hs[b];
    const uint8_t* src_b = src + src_offsets[b];
    float* dst_b = dst + static_cast<size_t>(b) * 3 * dst_h * dst_w;
    const int plane = dst_h * dst_w;
    const int dst_idx = static_cast<int>(y) * dst_w + static_cast<int>(x);

    const float src_xf = (static_cast<float>(x) - pad_w) / scale;
    const float src_yf = (static_cast<float>(y) - pad_h) / scale;
    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        dst_b[0 * plane + dst_idx] = pad_value * scrfd_alpha[0] + scrfd_beta[0];
        dst_b[1 * plane + dst_idx] = pad_value * scrfd_alpha[1] + scrfd_beta[1];
        dst_b[2 * plane + dst_idx] = pad_value * scrfd_alpha[2] + scrfd_beta[2];
        return;
    }
    const int src_idx = (src_y * src_w + src_x) * 3;
    const float b0 = src_b[src_idx + 0];
    const float g = src_b[src_idx + 1];
    const float r = src_b[src_idx + 2];
    dst_b[0 * plane + dst_idx] = r * scrfd_alpha[0] + scrfd_beta[0];
    dst_b[1 * plane + dst_idx] = g * scrfd_alpha[1] + scrfd_beta[1];
    dst_b[2 * plane + dst_idx] = b0 * scrfd_alpha[2] + scrfd_beta[2];
}

struct ScrfdPreprocWorkspace {
    uint8_t* d_src = nullptr;
    size_t capacity = 0;

    ~ScrfdPreprocWorkspace() {
        if (d_src) cudaFree(d_src);
    }
};

static thread_local ScrfdPreprocWorkspace scrfd_ws0;
static thread_local ScrfdPreprocWorkspace scrfd_ws1;

// SCRFD 整批参数数组池：一次 cudaMalloc 打包全部 kernel 参数，跨调用复用
struct ScrfdBatchParamWorkspace {
    uint8_t* d_ptr = nullptr;
    size_t capacity = 0;

    ~ScrfdBatchParamWorkspace() {
        if (d_ptr) cudaFree(d_ptr);
    }
};
static thread_local ScrfdBatchParamWorkspace scrfd_param_ws;

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

    bool scrfd_preprocess_cuda(
        const ImageData& image,
        Tensor* output,
        const std::vector<int>& dst_size,
        const float pad_val,
        LetterBoxRecord* letter_box_record,
        cudaStream_t stream,
        CudaOutputBufferPool* dst_pool) {
        return scrfd_preprocess_bgr_cuda(image.plane(0).data,
                                         {image.width(), image.height()},
                                         output,
                                         dst_size,
                                         pad_val,
                                         letter_box_record,
                                         stream,
                                         dst_pool);
    }

    bool scrfd_preprocess_bgr_cuda(const uint8_t* src,
                                   const std::vector<int>& src_size,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   const float pad_val,
                                   LetterBoxRecord* letter_box_record,
                                   cudaStream_t stream,
                                   CudaOutputBufferPool* dst_pool) {
        if (!output || dst_size.size() != 2) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        // 输出：从缓冲池获取，零拷贝包装
        float* dst_ptr = wrap_output_tensor(output, dst_pool, {3, dst_h, dst_w},
                                            DataType::FP32, output->get_name());
        if (!dst_ptr) return false;

        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }

        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});

        const size_t src_bytes = static_cast<size_t>(src_h) * src_w * 3;
        const uint8_t* d_src = nullptr;
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        if (is_device) {
            d_src = src;
        } else {
            if (scrfd_ws0.capacity < src_bytes) {
                if (scrfd_ws0.d_src) cudaFree(scrfd_ws0.d_src);
                cudaMalloc(&scrfd_ws0.d_src, src_bytes);
                scrfd_ws0.capacity = src_bytes;
            }
            cudaMemcpyAsync(scrfd_ws0.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
            d_src = scrfd_ws0.d_src;
        }

        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        scrfd_kernel_bgr_fusion<<<grid, block, 0, stream>>>(
            d_src, src_h, src_w,
            dst_ptr, dst_h, dst_w,
            letter_box_record->scale, letter_box_record->pad_w, letter_box_record->pad_h,
            pad_val);

        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) return false;

        output->expand_dim(0);
        return true;
    }

    bool scrfd_preprocess_nv12_cuda(const uint8_t* src_y,
                                    const uint8_t* src_uv,
                                    const std::vector<int>& src_size,
                                    const int step_y,
                                    const int step_uv,
                                    Tensor* output,
                                    const std::vector<int>& dst_size,
                                    const float pad_value,
                                    LetterBoxRecord* letter_box_record,
                                    cudaStream_t stream,
                                    CudaOutputBufferPool* dst_pool) {
        if (!output || dst_size.size() != 2) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        // 输出：从缓冲池获取，零拷贝包装
        float* dst_ptr = wrap_output_tensor(output, dst_pool, {3, dst_h, dst_w},
                                            DataType::FP32, output->get_name());
        if (!dst_ptr) return false;

        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }

        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});

        const size_t src_y_bytes = static_cast<size_t>(src_h) * step_y;
        const size_t src_uv_bytes = static_cast<size_t>(src_h / 2) * step_uv;
        const uint8_t* d_src_y = nullptr;
        const uint8_t* d_src_uv = nullptr;
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src_y) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        if (is_device) {
            d_src_y = src_y;
            d_src_uv = src_uv;
        } else {
            if (scrfd_ws0.capacity < src_y_bytes) {
                if (scrfd_ws0.d_src) cudaFree(scrfd_ws0.d_src);
                cudaMalloc(&scrfd_ws0.d_src, src_y_bytes);
                scrfd_ws0.capacity = src_y_bytes;
            }
            cudaMemcpyAsync(scrfd_ws0.d_src, src_y, src_y_bytes, cudaMemcpyHostToDevice, stream);
            d_src_y = scrfd_ws0.d_src;

            if (scrfd_ws1.capacity < src_uv_bytes) {
                if (scrfd_ws1.d_src) cudaFree(scrfd_ws1.d_src);
                cudaMalloc(&scrfd_ws1.d_src, src_uv_bytes);
                scrfd_ws1.capacity = src_uv_bytes;
            }
            cudaMemcpyAsync(scrfd_ws1.d_src, src_uv, src_uv_bytes, cudaMemcpyHostToDevice, stream);
            d_src_uv = scrfd_ws1.d_src;
        }

        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        scrfd_kernel_nv12_fusion<<<grid, block, 0, stream>>>(
            d_src_y, d_src_uv, src_h, src_w,
            step_y, step_uv,
            dst_ptr, dst_h, dst_w,
            letter_box_record->scale, letter_box_record->pad_w, letter_box_record->pad_h,
            pad_value);

        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) return false;

        output->expand_dim(0);
        return true;
    }

    bool scrfd_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                     Tensor* output,
                                     const std::vector<int>& dst_size,
                                     const float pad_value,
                                     std::vector<LetterBoxRecord>* letter_box_records,
                                     CudaOutputBufferPool* dst_pool,
                                     cudaStream_t stream) {
        if (images.empty() || dst_size.size() != 2 || !output) return false;
        const int batch = static_cast<int>(images.size());
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        // 整批输出：从缓冲池获取（复用），零拷贝包装
        float* dst_ptr = wrap_output_tensor(output, dst_pool, {batch, 3, dst_h, dst_w},
                                            DataType::FP32, output->get_name());
        if (!dst_ptr) return false;

        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }

        std::vector<int> src_ws(batch), src_hs(batch);
        std::vector<size_t> src_offsets(batch);
        std::vector<float> scales(batch), pad_ws(batch), pad_hs(batch);
        letter_box_records->resize(batch);
        size_t total_src_bytes = 0;
        for (int i = 0; i < batch; ++i) {
            const int sw = images[i].width(), sh = images[i].height();
            src_ws[i] = sw;
            src_hs[i] = sh;
            const LetterBoxRecord r = utils::cal_letter_box_param({sw, sh}, dst_size);
            (*letter_box_records)[i] = r;
            scales[i] = r.scale;
            pad_ws[i] = r.pad_w;
            pad_hs[i] = r.pad_h;
            src_offsets[i] = total_src_bytes;
            total_src_bytes += static_cast<size_t>(sh) * sw * 3;
        }

        // 源图拼接上传（thread_local 池复用）
        if (scrfd_ws0.capacity < total_src_bytes) {
            if (scrfd_ws0.d_src) cudaFree(scrfd_ws0.d_src);
            if (cudaMalloc(&scrfd_ws0.d_src, total_src_bytes) != cudaSuccess) return false;
            scrfd_ws0.capacity = total_src_bytes;
        }
        for (int i = 0; i < batch; ++i) {
            cudaMemcpyAsync(scrfd_ws0.d_src + src_offsets[i], images[i].plane(0).data,
                            static_cast<size_t>(src_hs[i]) * src_ws[i] * 3,
                            cudaMemcpyHostToDevice, stream);
        }

        // 参数数组单块打包 + 池复用
        const size_t need = sizeof(size_t) * batch + sizeof(int) * batch * 2 + sizeof(float) * batch * 3;
        if (scrfd_param_ws.capacity < need) {
            if (scrfd_param_ws.d_ptr) cudaFree(scrfd_param_ws.d_ptr);
            if (cudaMalloc(&scrfd_param_ws.d_ptr, need) != cudaSuccess) return false;
            scrfd_param_ws.capacity = need;
        }
        uint8_t* base = scrfd_param_ws.d_ptr;
        auto* d_offsets = reinterpret_cast<size_t*>(base);
        auto* d_ws = reinterpret_cast<int*>(base + sizeof(size_t) * batch);
        auto* d_hs = reinterpret_cast<int*>(base + sizeof(size_t) * batch + sizeof(int) * batch);
        auto* d_scales = reinterpret_cast<float*>(base + sizeof(size_t) * batch + sizeof(int) * batch * 2);
        auto* d_pad_ws = d_scales + batch;
        auto* d_pad_hs = d_scales + batch * 2;
        cudaMemcpyAsync(d_offsets, src_offsets.data(), sizeof(size_t) * batch, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ws, src_ws.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_hs, src_hs.data(), sizeof(int) * batch, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_scales, scales.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pad_ws, pad_ws.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pad_hs, pad_hs.data(), sizeof(float) * batch, cudaMemcpyHostToDevice, stream);

        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch);
        scrfd_kernel_bgr_fusion_batch<<<grid, block, 0, stream>>>(
            scrfd_ws0.d_src, d_ws, d_hs, d_offsets, d_scales, d_pad_ws, d_pad_hs,
            dst_ptr, dst_h, dst_w, pad_value);

        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        return err == cudaSuccess;
    }
} // namespace modeldeploy::vision