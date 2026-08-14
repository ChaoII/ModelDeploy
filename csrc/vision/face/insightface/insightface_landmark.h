//
// insightface buffalo_l landmark：2d106det（2D 106 点）+ 1k3d68（3D 68 点 + 姿态）。
// 标准架构：Preprocessor（fused_preprocess center-crop）→ Runtime → Postprocessor（逆仿射 + 姿态）。
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
#include "vision/face/insightface/face_align_utils.h"

namespace modeldeploy::vision::face {

    // 前处理：以 bbox 中心 transform（scale=192/(max(w,h)*1.5)）裁剪到 192x192。
    // 模型输入为 0-255 RGB（模型内部归一化），仅 swapRB。
    // 映射到 fused_preprocess：origin = output/2 - center*scale, scale 各轴同。
    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmarkPreprocessor {
    public:
        InsightFaceLandmarkPreprocessor();

        // 单人脸：输出 [1,3,192,192] FP32（0-255，模型内部归一化）
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

    // 后处理：landmark 输出 (+1)*scale -> 逆仿射回原图；3D 额外姿态估计
    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmarkPostprocessor {
    public:
        // 2D 106 点
        bool run_2d(const std::vector<Tensor>& infer_results, const cv::Mat& inv_M,
                    const int input_size, std::vector<std::array<float, 2>>* landmarks);
        // 3D 68 点 + 姿态
        bool run_3d(const std::vector<Tensor>& infer_results, const cv::Mat& inv_M,
                    const int input_size, std::vector<std::array<float, 3>>* landmarks,
                    std::array<float, 3>* pose);
    };

    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmark : public BaseModel {
    public:
        explicit InsightFaceLandmark(const std::string& model_file,
                                     const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceLandmark"; }

        bool predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                           std::vector<std::array<float, 2>>* landmarks,
                           TimerArray* timers = nullptr);
        bool predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                          std::vector<std::array<float, 3>>* landmarks,
                          std::array<float, 3>* pose,
                          TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceLandmark> clone() const;

        virtual InsightFaceLandmarkPreprocessor& get_preprocessor() { return preprocessor_; }
        virtual InsightFaceLandmarkPostprocessor& get_postprocessor() { return postprocessor_; }

        std::vector<int> input_size_{192, 192};

    protected:
        bool Initialize();
        InsightFaceLandmarkPreprocessor preprocessor_;
        InsightFaceLandmarkPostprocessor postprocessor_;
    };

} // namespace modeldeploy::vision::face
