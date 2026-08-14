#include "vision/processors/cuda/bgr_to_nv12.cuh"

namespace modeldeploy::vision {
    // BT.709 limited range：Y = (66R + 129G + 25B + 128)>>8 + 16
    __global__ void kernel_bgr_to_nv12_y(const uint8_t* __restrict__ bgr,
                                         uint8_t* __restrict__ y, int w, int h) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y_ = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y_ >= h) return;
        const int idx = (y_ * w + x) * 3;
        const int r = bgr[idx + 2], g = bgr[idx + 1], b = bgr[idx + 0];
        y[y_ * w + x] = (uint8_t)(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
    }

    // UV：2x2 平均后按 BT.709 limited range 计算，interleaved (y*uvw + x)*2，偏移在 Y 平面之后
    __global__ void kernel_bgr_to_nv12_uv(const uint8_t* __restrict__ bgr,
                                          uint8_t* __restrict__ uv, int w, int h) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y_ = blockIdx.y * blockDim.y + threadIdx.y;
        const int uvw = w / 2, uvh = h / 2;
        if (x >= uvw || y_ >= uvh) return;
        int r = 0, g = 0, b = 0;
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const int idx = ((y_ * 2 + dy) * w + (x * 2 + dx)) * 3;
            r += bgr[idx + 2]; g += bgr[idx + 1]; b += bgr[idx + 0];
        }
        r >>= 2; g >>= 2; b >>= 2;
        const int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        const int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uint8_t* p = uv + (y_ * uvw + x) * 2;
        p[0] = (uint8_t)u; p[1] = (uint8_t)v;
    }

    struct BgrToNv12Workspace {
        uint8_t* d_bgr = nullptr;
        uint8_t* d_nv12 = nullptr;
        size_t bgr_capacity = 0;
        size_t nv12_capacity = 0;

        ~BgrToNv12Workspace() {
            if (d_bgr) cudaFree(d_bgr);
            if (d_nv12) cudaFree(d_nv12);
        }
    };

    static thread_local BgrToNv12Workspace bgr_nv12_ws;

    bool bgr_to_nv12_cuda(const uint8_t* bgr, int width, int height,
                          uint8_t* nv12, cudaStream_t stream) {
        if (!bgr || !nv12 || width <= 0 || height <= 0) return false;

        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }

        const size_t bgr_bytes = static_cast<size_t>(height) * width * 3;
        const size_t y_bytes = static_cast<size_t>(height) * width;
        const size_t nv12_bytes = bgr_bytes / 2;

        // 内部 stream 上可能已排队异步操作：失败时先同步再销毁，避免悬挂操作
        auto fail = [&]() {
            if (is_internal_stream) {
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
            return false;
        };

        // 输入 BGR：device 指针直接用，host 指针自动上传
        cudaPointerAttributes bgr_attr{};
        const bool bgr_on_device =
            cudaPointerGetAttributes(&bgr_attr, bgr) == cudaSuccess &&
            bgr_attr.type == cudaMemoryTypeDevice;

        const uint8_t* d_bgr;
        if (bgr_on_device) {
            d_bgr = bgr;
        } else {
            if (bgr_nv12_ws.bgr_capacity < bgr_bytes) {
                uint8_t* new_buf = nullptr;
                if (cudaMalloc(&new_buf, bgr_bytes) != cudaSuccess) return fail();
                if (bgr_nv12_ws.d_bgr) cudaFree(bgr_nv12_ws.d_bgr);
                bgr_nv12_ws.d_bgr = new_buf;
                bgr_nv12_ws.bgr_capacity = bgr_bytes;
            }
            if (cudaMemcpyAsync(bgr_nv12_ws.d_bgr, bgr, bgr_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
                return fail();
            d_bgr = bgr_nv12_ws.d_bgr;
        }

        // 输出 NV12：device 指针直接用；host 指针先写到 GPU 缓冲再回拷
        cudaPointerAttributes nv12_attr{};
        const bool nv12_on_device =
            cudaPointerGetAttributes(&nv12_attr, nv12) == cudaSuccess &&
            nv12_attr.type == cudaMemoryTypeDevice;

        uint8_t* d_nv12;
        if (nv12_on_device) {
            d_nv12 = nv12;
        } else {
            if (bgr_nv12_ws.nv12_capacity < nv12_bytes) {
                uint8_t* new_buf = nullptr;
                if (cudaMalloc(&new_buf, nv12_bytes) != cudaSuccess) return fail();
                if (bgr_nv12_ws.d_nv12) cudaFree(bgr_nv12_ws.d_nv12);
                bgr_nv12_ws.d_nv12 = new_buf;
                bgr_nv12_ws.nv12_capacity = nv12_bytes;
            }
            d_nv12 = bgr_nv12_ws.d_nv12;
        }

        dim3 block(16, 16);
        dim3 grid_y((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        kernel_bgr_to_nv12_y<<<grid_y, block, 0, stream>>>(d_bgr, d_nv12, width, height);

        dim3 grid_uv(((width / 2) + block.x - 1) / block.x, ((height / 2) + block.y - 1) / block.y);
        kernel_bgr_to_nv12_uv<<<grid_uv, block, 0, stream>>>(
            d_bgr, d_nv12 + y_bytes, width, height);

        cudaError_t launch_err = cudaGetLastError();
        cudaError_t copy_err = cudaSuccess;
        if (!nv12_on_device) {
            copy_err = cudaMemcpyAsync(nv12, d_nv12, nv12_bytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        return launch_err == cudaSuccess && copy_err == cudaSuccess && sync_err == cudaSuccess;
    }
}
