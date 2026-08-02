//
// Created by aichao on 2025/7/22.
//

#include "vision/common/processors/fusion_resize_pad_normalize_permute.h"
#include "vision/processors/cpu/simd/fused_preproc_simd.h"


namespace modeldeploy::vision {
    bool fusion_resize_pad_normalize_permute_cpu(
        const std::vector<ImageData>& images, Tensor* output,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const float pad_value) {
        const int batch_size = static_cast<int>(images.size());
        if (batch_size == 0) return false;

        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        const float alpha[3] = {
            1.0f / 255.0f / std[0],
            1.0f / 255.0f / std[1],
            1.0f / 255.0f / std[2]
        };
        const float beta[3] = {
            -mean[0] / std[0],
            -mean[1] / std[1],
            -mean[2] / std[2]
        };
        // pad 区为仿射后空间（逐通道）
        const float pad[3] = {
            pad_value * alpha[0] + beta[0],
            pad_value * alpha[1] + beta[1],
            pad_value * alpha[2] + beta[2]
        };

        output->allocate({batch_size, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = output->data_ptr<float>();
        // 运行时 dispatch：标量/AVX2/AVX512/NEON/SVE，每图独立 resize 尺寸
        const auto kernel = get_fusion_rpnp_kernel();
        for (int i = 0; i < batch_size; ++i) {
            kernel(images[i].data(), images[i].width(), images[i].height(),
                   dst + static_cast<size_t>(i) * 3 * dst_h * dst_w,
                   dst_w, dst_h, resize_sizes[i][0], resize_sizes[i][1],
                   alpha, beta, pad);
        }
        return true;
    }
}
