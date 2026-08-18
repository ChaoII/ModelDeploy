//
// Created by aichao on 2025/06/2.
//
#pragma once


#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/pose/preprocessor.h"
#include "vision/pose/postprocessor.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPose : public BaseModel {
    public:
        explicit UltralyticsPose(const std::string& model_file,
                                 const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsPose"; }

        bool predict(const ImageData& image, std::vector<KeyPointsResult>* result, TimerArray* timers = nullptr);

        bool batch_predict(const std::vector<ImageData>& images,
                            std::vector<std::vector<KeyPointsResult>>* results, TimerArray* timers = nullptr);
        bool predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                          int width, int height, int step_y, int step_uv,
                          std::vector<KeyPointsResult>* result, LetterBoxRecord* letter_box_record = nullptr,
                          ImageData* out_frame = nullptr,
                          Device src_device = Device::CPU,
                          TimerArray* timers = nullptr);

        // 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）
        bool draw_result(ImageData& frame, const std::vector<KeyPointsResult>& result,
                         double threshold = 0.5);


        [[nodiscard]] std::unique_ptr<UltralyticsPose> clone() const;


        UltralyticsPosePreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        UltralyticsPosePostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        // NV12/device 单帧共享推理核心（预处走 plane(0)/plane(1) 零拷贝），predict/predict_nv12 共用
        bool predict_single_nv12(const ImageData& frame,
                                 std::vector<KeyPointsResult>* result,
                                 LetterBoxRecord* letter_box_record,
                                 TimerArray* timers);
        UltralyticsPosePreprocessor preprocessor_;
        UltralyticsPosePostprocessor postprocessor_;
    };
} // namespace detection
