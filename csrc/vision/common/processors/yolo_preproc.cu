#include "vision/common/processors/yolo_preproc.cuh"
#include <cuda_runtime.h>
#include <vision/utils.h>

// RGB 三通道归一化均值和标准差
__constant__ float k_mean[3] = {0.f, 0.f, 0.f};
__constant__ float k_std[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

__global__ void kernel_bgr_fusion(
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

    // 边界检查，并不一定线程数和像素的数量一致，有可能线程数量大于输出的像素数量
    if (x >= dst_w || y >= dst_h) return;

    const float src_xf = (x - pad_w) / scale;
    const float src_yf = (y - pad_h) / scale;

    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    const int dst_idx = y * dst_w + x;
    const int plane_size = dst_h * dst_w;

    // 这个范围内的像素都padding值
    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        const float v0 = (pad_value - k_mean[0]) * k_std[0];
        const float v1 = (pad_value - k_mean[1]) * k_std[1];
        const float v2 = (pad_value - k_mean[2]) * k_std[2];
        dst[0 * plane_size + dst_idx] = v0;
        dst[1 * plane_size + dst_idx] = v1;
        dst[2 * plane_size + dst_idx] = v2;
    }
    else {
        // [B0G0R0 B1G1R1 B2G2R2 B3G3R3]
        // [B4G4R4 B5G5R5 B6G6R6 B7G7R7]
        // [B8G8R8 B9G9R9 ...... ......]

        // [B0B1B2B3B4B5B6B7B8B9...]
        // [G0G1G2G3G4G5G6G7G8G9...]
        // [R0R1R2R3R4R5R6R7R8R9...]
        const int src_idx = (src_y * src_w + src_x) * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        dst[0 * plane_size + dst_idx] = (r - k_mean[0]) * k_std[0];
        dst[1 * plane_size + dst_idx] = (g - k_mean[1]) * k_std[1];
        dst[2 * plane_size + dst_idx] = (b - k_mean[2]) * k_std[2];
    }
}

/**
 * @brief NV12 融合核函数: NV12 -> RGB (CHW) + Resize (最近邻) + Letterbox + Normalize
 *
 * @param srcY      [In] Y 平面显存指针
 * @param srcUV     [In] UV 平面显存指针 (交错存储 UVUV...)
 * @param src_h     [In] 原始图像高度
 * @param src_w     [In] 原始图像宽度
 * @param stepY     [In] Y 平面行步长 (bytes)
 * @param stepUV    [In] UV 平面行步长 (bytes)
 * @param dst       [Out] 输出显存指针 [3, dst_h, dst_w] (Float)
 * @param dst_h     [In] 目标高度
 * @param dst_w     [In] 目标宽度
 * @param scale     [In] 缩放比例 (resize_w / src_w 或 resize_h / src_h，取较小者)
 * @param pad_w     [In] 左侧填充像素数 (offsetX)
 * @param pad_h     [In] 上侧填充像素数 (offsetY)
 * @param pad_value [In] 填充值 (例如 114.0)
 */
__global__ void kernel_nv12_fusion(
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

    // 边界检查
    if (x >= dst_w || y >= dst_h) return;

    // 反推源图像坐标 (最近邻插值)
    // dst_coord = pad + src_coord * scale  =>  src_coord = (dst_coord - pad) / scale
    const float src_xf = (x - pad_w) / scale;
    const float src_yf = (y - pad_h) / scale;

    const int src_x = static_cast<int>(src_xf);
    const int src_y = static_cast<int>(src_yf);

    const int dst_idx = y * dst_w + x;
    const int plane_size = dst_h * dst_w;

    // 判断是否在有效图像区域内 (Letterbox 逻辑)
    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        // --- 填充区域 ---
        const float v0 = (pad_value - k_mean[0]) * k_std[0];
        const float v1 = (pad_value - k_mean[1]) * k_std[1];
        const float v2 = (pad_value - k_mean[2]) * k_std[2];
        // 写入 CHW: C0(R), C1(G), C2(B)
        dst[0 * plane_size + dst_idx] = v0;
        dst[1 * plane_size + dst_idx] = v1;
        dst[2 * plane_size + dst_idx] = v2;
    }
    else {
        // --- 有效图像区域 ---

        // 1. 采样 Y 分量
        const float y_val = srcY[src_y * stepY + src_x];

        // 2. 采样 U, V 分量
        // NV12: UV 平面分辨率是 Y 的一半，且交错存储 (U0, V0, U1, V1...)
        const int uv_x = src_x >> 1; // src_x / 2
        const int uv_y = src_y >> 1; // src_y / 2

        // 边界保护 (防止 src_w/src_h 为奇数时越界)
        const int safe_uv_x = min(uv_x, (src_w >> 1) - 1);
        const int safe_uv_y = min(uv_y, (src_h >> 1) - 1);

        const uint8_t* uv_row = srcUV + safe_uv_y * stepUV;
        float u_val = uv_row[safe_uv_x * 2 + 0]; // U
        float v_val = uv_row[safe_uv_x * 2 + 1]; // V

        // 3. YUV -> RGB 转换 (BT.601 标准)
        // Y 范围 0-255, UV 范围 0-255 (中心 128)
        u_val -= 128.0f;
        v_val -= 128.0f;

        // 计算 RGB (结果范围 0-255)
        float r = y_val + 1.402f * v_val;
        float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
        float b = y_val + 1.772f * u_val;

        // clip到 [0, 255] 防止溢出
        r = fminf(fmaxf(r, 0.0f), 255.0f);
        g = fminf(fmaxf(g, 0.0f), 255.0f);
        b = fminf(fmaxf(b, 0.0f), 255.0f);

        // 4. 归一化/标准化 并写入 CHW
        dst[0 * plane_size + dst_idx] = (r - k_mean[0]) * k_std[0]; // R -> C0
        dst[1 * plane_size + dst_idx] = (g - k_mean[1]) * k_std[1]; // G -> C1
        dst[2 * plane_size + dst_idx] = (b - k_mean[2]) * k_std[2]; // B -> C2
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

static thread_local PreprocWorkspace ws0;
static thread_local PreprocWorkspace ws1;

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(
        const ImageData& image,
        Tensor* output,
        const std::vector<int>& dst_size,
        const float pad_val,
        LetterBoxRecord* letter_box_record,
        cudaStream_t stream) {
        return yolo_preprocess_bgr_cuda(image.data(),
                                        {image.width(), image.height()},
                                        output,
                                        dst_size,
                                        pad_val,
                                        letter_box_record,
                                        stream);
    }

    bool yolo_preprocess_bgr_cuda(const uint8_t* src,
                                  const std::vector<int>& src_size,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  const float pad_val,
                                  LetterBoxRecord* letter_box_record,
                                  cudaStream_t stream) {
        if (!output || dst_size.size() != 2) {
            return false;
        }
        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        // 1 output: GPU, FP32, CHW
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

        // 2 CUDA stream
        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }
        // 3 letterbox（host 计算）
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        // BGR
        const size_t src_bytes = static_cast<size_t>(src_h) * src_w * 3;
        const uint8_t* d_src = nullptr;
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        if (is_device) {
            d_src = src;
        }
        else {
            if (ws0.capacity < src_bytes) {
                if (ws0.d_src) cudaFree(ws0.d_src);
                cudaMalloc(&ws0.d_src, src_bytes);
                ws0.capacity = src_bytes;
            }
            cudaMemcpyAsync(ws0.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
            d_src = ws0.d_src;
        }
        // 5 launch kernel
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        kernel_bgr_fusion<<<grid, block, 0, stream>>>(
            d_src,
            src_h,
            src_w,
            output->data_ptr<float>(),
            dst_h,
            dst_w,
            letter_box_record->scale,
            letter_box_record->pad_w,
            letter_box_record->pad_h,
            pad_val);
        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            return false;
        }
        // 6 增加batch维
        output->expand_dim(0);
        return true;
    }

    bool yolo_preprocess_nv12_cuda(const uint8_t* src_y,
                                   const uint8_t* src_uv,
                                   const std::vector<int>& src_size,
                                   const int step_y,
                                   const int step_uv,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   const float pad_value,
                                   LetterBoxRecord* letter_box_record,
                                   cudaStream_t stream) {
        if (!output || dst_size.size() != 2) {
            return false;
        }
        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        // 1 output: GPU, FP32, CHW
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

        // 2 CUDA stream
        bool is_internal_stream = false;
        if (stream == nullptr) {
            if (cudaStreamCreate(&stream) != cudaSuccess) return false;
            is_internal_stream = true;
        }
        // 3 letterbox（host 计算）
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        // NV12
        // 使用 step_y 计算实际占用字节更稳妥，或者 src_h * src_w
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
        }
        else {
            if (ws0.capacity < src_y_bytes) {
                if (ws0.d_src) cudaFree(ws0.d_src);
                cudaMalloc(&ws0.d_src, src_y_bytes);
                ws0.capacity = src_y_bytes;
            }
            cudaMemcpyAsync(ws0.d_src, src_y, src_y_bytes, cudaMemcpyHostToDevice, stream);
            d_src_y = ws0.d_src;

            if (ws1.capacity < src_uv_bytes) {
                if (ws1.d_src) cudaFree(ws1.d_src);
                cudaMalloc(&ws1.d_src, src_uv_bytes);
                ws1.capacity = src_uv_bytes;
            }
            cudaMemcpyAsync(ws1.d_src, src_uv, src_uv_bytes, cudaMemcpyHostToDevice, stream);
            d_src_uv = ws1.d_src;
        }
        // 5 launch kernel
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        kernel_nv12_fusion<<<grid, block, 0, stream>>>(
            d_src_y,
            d_src_uv,
            src_h,
            src_w,
            step_y,
            step_uv,
            output->data_ptr<float>(),
            dst_h,
            dst_w,
            letter_box_record->scale,
            letter_box_record->pad_w,
            letter_box_record->pad_h,
            pad_value);
        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        if (is_internal_stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            return false;
        }
        // 6 增加batch维
        output->expand_dim(0);
        return true;
    }
} // namespace modeldeploy::vision
