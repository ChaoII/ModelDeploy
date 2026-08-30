#pragma once
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/sam/preprocessor.h"
#include "vision/sam/postprocessor.h"

namespace modeldeploy::vision::seg {
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

        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<std::vector<InstanceSegResult>>* results,
                           TimerArray* timers = nullptr);

        /// 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）
        bool draw_result(ImageData& frame, const std::vector<InstanceSegResult>& result,
                         double threshold = 0.3);

        [[nodiscard]] std::unique_ptr<FastSam> clone() const;

        FastSamPreprocessor& get_preprocessor() { return *preprocessor_; }
        FastSamPostprocessor& get_postprocessor() { return *postprocessor_; }

    protected:
        bool initialize();
        std::unique_ptr<FastSamPreprocessor> preprocessor_;
        std::unique_ptr<FastSamPostprocessor> postprocessor_;
    };
} // namespace modeldeploy::vision::seg
