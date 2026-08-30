#pragma once
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/sam/preprocessor.h"
#include "vision/sam/postprocessor.h"

namespace modeldeploy::vision::seg {
    /*! @brief FastSAM 交互式提示集合（bbox / point），用于 predict_with_prompts 过滤实例。
     */
    struct MODELDEPLOY_CXX_EXPORT FastSamPrompts {
        std::vector<Rect2f> bboxes;        // (x, y, w, h), 原图像素
        std::vector<Point2f> points;       // 原图像素
        std::vector<int> point_labels;     // 与 points 等长; 1=前景, 0=背景
        [[nodiscard]] bool empty() const { return bboxes.empty() && points.empty(); }
    };

    /*! @brief FastSAM 轻量分割一切模型。
     *  一次性输出 box + mask（对齐 UltralyticsSeg 消费路径），后处理产出 InstanceSegResult。
     */
    class MODELDEPLOY_CXX_EXPORT FastSam : public BaseModel {
    public:
        explicit FastSam(const std::string& model_file,
                         const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "FastSam"; }

        bool predict(const ImageData& image, std::vector<InstanceSegResult>* result,
                     TimerArray* timers = nullptr);

        /// 在原 predict 基础上按 prompts（bbox 取最大 IoU 实例、point 按掩码命中保留/剔除）过滤实例。
        bool predict_with_prompts(const ImageData& image, const FastSamPrompts& prompts,
                                  std::vector<InstanceSegResult>* result, TimerArray* timers = nullptr);

        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<std::vector<InstanceSegResult>>* results,
                           TimerArray* timers = nullptr);

        /// 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）
        bool draw_result(ImageData& frame, const std::vector<InstanceSegResult>& result,
                         double threshold = 0.3);

        [[nodiscard]] std::unique_ptr<FastSam> clone() const;

        FastSamPreprocessor& get_preprocessor() { return preprocessor_; }
        FastSamPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        FastSamPreprocessor preprocessor_;
        FastSamPostprocessor postprocessor_;
    };
} // namespace modeldeploy::vision::seg
