//
// insightface buffalo_l det_10g 前处理。
// 复用 VisionProcessorBackend（fused_preprocess 最近邻），多后端天然支持。
// 与 python insightface SCRFD._detect_candidates 对齐：等比例缩放 + 左上放置 + pad 0。
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceDetPreprocessor {
    public:
        InsightFaceDetPreprocessor();

        // 单图：输出 [1,3,H,W] FP32（由 backend 决定 CPU/GPU）
        bool run(const ImageData& image, Tensor* output,
                 LetterBoxRecord* letter_box_record) const;

        // 整批：输出 [B,3,H,W]
        bool run(const std::vector<ImageData>& images, Tensor* output,
                 std::vector<LetterBoxRecord>* letter_box_records) const;

        void set_size(const std::vector<int>& size) { size_ = size; }
        [[nodiscard]] std::vector<int> get_size() const { return size_; }

        void set_processor_backend(std::shared_ptr<VisionProcessorBackend> backend) {
            backend_ = std::move(backend);
        }
        [[nodiscard]] std::shared_ptr<VisionProcessorBackend> get_processor_backend() const {
            return backend_;
        }

        float det_thresh = 0.5f;

    private:
        std::vector<int> size_{640, 640};
        std::shared_ptr<VisionProcessorBackend> backend_ =
            std::make_shared<CpuProcessorBackend>();
    };

} // namespace modeldeploy::vision::face
