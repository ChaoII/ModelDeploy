//
// insightface buffalo_l det_10g：SCRFD 人脸检测。
// 标准架构：InsightFaceDetPreprocessor（fused_preprocess，多后端）→ Runtime → InsightFaceDetPostprocessor（解码）。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/face/insightface/insightface_types.h"

namespace modeldeploy::vision::face {

    // 前处理：letterbox（左上放置）→ (x-127.5)/128 → BGR2RGB → CHW
    // 与 python insightface SCRFD._detect_candidates 逐值对齐。
    class MODELDEPLOY_CXX_EXPORT InsightFaceDetPreprocessor {
    public:
        InsightFaceDetPreprocessor();

        // 单图：输出 [1,3,H,W] FP32 GPU/CPU（由 backend 决定）
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

    // 后处理：stride 8/16/32 双 anchor 解码 + distance2bbox/kps + 缩放回原图 + NMS
    class MODELDEPLOY_CXX_EXPORT InsightFaceDetPostprocessor {
    public:
        bool run(const std::vector<Tensor>& infer_results,
                 const std::vector<LetterBoxRecord>& letter_box_records,
                 const std::vector<float>& det_scales,
                 std::vector<std::vector<InsightFaceBox>>* results);

        float nms_thresh_ = 0.4f;
    };

    class MODELDEPLOY_CXX_EXPORT InsightFaceDet : public BaseModel {
    public:
        explicit InsightFaceDet(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceDet"; }

        // 检测：输入 BGR 图，输出人脸框（原图坐标）+ 5 关键点（原图坐标）
        bool predict(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                     TimerArray* timers = nullptr);
        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<std::vector<InsightFaceBox>>* boxes,
                           TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceDet> clone() const;

        virtual InsightFaceDetPreprocessor& get_preprocessor() { return preprocessor_; }
        virtual InsightFaceDetPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        InsightFaceDetPreprocessor preprocessor_;
        InsightFaceDetPostprocessor postprocessor_;
    };

} // namespace modeldeploy::vision::face
