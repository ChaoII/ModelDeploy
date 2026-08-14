//
// insightface buffalo_l landmark 前处理。
// 复用 VisionProcessorBackend（fused_preprocess），多后端天然支持。
// 以 bbox 中心 transform（scale=192/(max(w,h)*1.5)）裁剪到 192x192。
//
#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmarkPreprocessor {
    public:
        InsightFaceLandmarkPreprocessor();

        // 单人脸：输出 [1,3,192,192] FP32（0-255，模型内部归一化）；M 为前向仿射矩阵
        bool run(const ImageData& image, const std::array<float, 4>& bbox,
                 cv::Mat* M, Tensor* output) const;

        void set_size(const std::vector<int>& size) { size_ = size; }
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }
        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

    private:
        std::vector<int> size_{192, 192};
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
    };

} // namespace modeldeploy::vision::face
