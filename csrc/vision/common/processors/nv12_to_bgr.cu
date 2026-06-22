#include "vision/common/processors/nv12_to_bgr.cuh"
#include <cuda_runtime.h>

namespace modeldeploy::vision {
    __global__ void kernel_nv12_to_bgr(
        const uint8_t* __restrict__ src_y,
        const uint8_t* __restrict__ src_uv,
        const int src_w,
        const int src_h,
        const int step_y,
        const int step_uv,
        uint8_t* __restrict__ dst_bgr) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= src_w || y >= src_h) return;

        const float y_val = static_cast<float>(src_y[y * step_y + x]);
        const int uv_x = x >> 1;
        const int uv_y = y >> 1;
        const int safe_uv_x = min(uv_x, (src_w >> 1) - 1);
        const int safe_uv_y = min(uv_y, (src_h >> 1) - 1);
        const uint8_t* uv_row = src_uv + safe_uv_y * step_uv;
        float u_val = static_cast<float>(uv_row[safe_uv_x * 2 + 0]) - 128.0f;
        float v_val = static_cast<float>(uv_row[safe_uv_x * 2 + 1]) - 128.0f;

        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
        float b = y_val + 1.772f * u_val;
        r = fminf(fmaxf(r, 0.0f), 255.0f);
        g = fminf(fmaxf(g, 0.0f), 255.0f);
        b = fminf(fmaxf(b, 0.0f), 255.0f);

        const int dst_idx = (y * src_w + x) * 3;
        dst_bgr[dst_idx + 0] = static_cast<uint8_t>(b);
        dst_bgr[dst_idx + 1] = static_cast<uint8_t>(g);
        dst_bgr[dst_idx + 2] = static_cast<uint8_t>(r);
    }

    struct Nv12ToBgrWorkspace {
        uint8_t* d_src_y = nullptr;
        uint8_t* d_src_uv = nullptr;
        uint8_t* d_dst_bgr = nullptr;
        size_t y_capacity = 0;
        size_t uv_capacity = 0;
        size_t bgr_capacity = 0;

        ~Nv12ToBgrWorkspace() {
            if (d_src_y) cudaFree(d_src_y);
            if (d_src_uv) cudaFree(d_src_uv);
            if (d_dst_bgr) cudaFree(d_dst_bgr);
        }
    };

    static thread_local Nv12ToBgrWorkspace nv12_bgr_ws;

    bool nv12_to_bgr_cuda(const uint8_t* src_y,
                           const uint8_t* src_uv,
                           int src_w, int src_h,
                           int step_y, int step_uv,
                           uint8_t* dst_bgr,
                           cudaStream_t stream) {
        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }

        const size_t y_bytes = static_cast<size_t>(src_h) * step_y;
        const size_t uv_bytes = static_cast<size_t>(src_h / 2) * step_uv;
        const size_t bgr_bytes = static_cast<size_t>(src_h) * src_w * 3;

        cudaPointerAttributes attr{};
        const bool src_is_device =
            cudaPointerGetAttributes(&attr, src_y) == cudaSuccess && attr.type == cudaMemoryTypeDevice;

        const uint8_t* d_y;
        const uint8_t* d_uv;
        uint8_t* d_dst;

        if (src_is_device) {
            d_y = src_y;
            d_uv = src_uv;
        } else {
            // Upload Y plane
            if (nv12_bgr_ws.y_capacity < y_bytes) {
                if (nv12_bgr_ws.d_src_y) cudaFree(nv12_bgr_ws.d_src_y);
                cudaMalloc(&nv12_bgr_ws.d_src_y, y_bytes);
                nv12_bgr_ws.y_capacity = y_bytes;
            }
            if (nv12_bgr_ws.uv_capacity < uv_bytes) {
                if (nv12_bgr_ws.d_src_uv) cudaFree(nv12_bgr_ws.d_src_uv);
                cudaMalloc(&nv12_bgr_ws.d_src_uv, uv_bytes);
                nv12_bgr_ws.uv_capacity = uv_bytes;
            }
            cudaMemcpyAsync(nv12_bgr_ws.d_src_y, src_y, y_bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(nv12_bgr_ws.d_src_uv, src_uv, uv_bytes, cudaMemcpyHostToDevice, stream);
            d_y = nv12_bgr_ws.d_src_y;
            d_uv = nv12_bgr_ws.d_src_uv;
        }

        // Allocate GPU output buffer for BGR
        if (nv12_bgr_ws.bgr_capacity < bgr_bytes) {
            if (nv12_bgr_ws.d_dst_bgr) cudaFree(nv12_bgr_ws.d_dst_bgr);
            cudaMalloc(&nv12_bgr_ws.d_dst_bgr, bgr_bytes);
            nv12_bgr_ws.bgr_capacity = bgr_bytes;
        }
        d_dst = nv12_bgr_ws.d_dst_bgr;

        dim3 block(16, 16);
        dim3 grid((src_w + block.x - 1) / block.x, (src_h + block.y - 1) / block.y);

        kernel_nv12_to_bgr<<<grid, block, 0, stream>>>(
            d_y, d_uv, src_w, src_h, step_y, step_uv, d_dst);

        // D2H copy for CPU output
        cudaPointerAttributes dst_attr{};
        bool dst_is_device =
            cudaPointerGetAttributes(&dst_attr, dst_bgr) == cudaSuccess && dst_attr.type == cudaMemoryTypeDevice;
        if (!dst_is_device) {
            cudaMemcpyAsync(dst_bgr, d_dst, bgr_bytes, cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(dst_bgr, d_dst, bgr_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        return err == cudaSuccess;
    }
}