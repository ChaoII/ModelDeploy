#include "vision/face/face_det/scrfd_preproc.cuh"
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

struct ScrfdPreprocWorkspace {
    uint8_t* d_src = nullptr;
    size_t capacity = 0;

    ~ScrfdPreprocWorkspace() {
        if (d_src) cudaFree(d_src);
    }
};

static thread_local ScrfdPreprocWorkspace scrfd_ws0;
static thread_local ScrfdPreprocWorkspace scrfd_ws1;

namespace modeldeploy::vision {
    bool scrfd_preprocess_cuda(
        const ImageData& image,
        Tensor* output,
        const std::vector<int>& dst_size,
        const float pad_val,
        LetterBoxRecord* letter_box_record,
        cudaStream_t stream) {
        return scrfd_preprocess_bgr_cuda(image.data(),
                                         {image.width(), image.height()},
                                         output,
                                         dst_size,
                                         pad_val,
                                         letter_box_record,
                                         stream);
    }

    bool scrfd_preprocess_bgr_cuda(const uint8_t* src,
                                   const std::vector<int>& src_size,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   const float pad_val,
                                   LetterBoxRecord* letter_box_record,
                                   cudaStream_t stream) {
        if (!output || dst_size.size() != 2) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

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
            output->data_ptr<float>(), dst_h, dst_w,
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
                                    cudaStream_t stream) {
        if (!output || dst_size.size() != 2) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

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
            output->data_ptr<float>(), dst_h, dst_w,
            letter_box_record->scale, letter_box_record->pad_w, letter_box_record->pad_h,
            pad_value);

        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) return false;

        output->expand_dim(0);
        return true;
    }
} // namespace modeldeploy::vision