//
// insightface buffalo_l det_10g：SCRFD 人脸检测模型。
// 标准架构（对齐 detection）：InsightFaceDetPreprocessor（backend）→ Runtime → InsightFaceDetPostprocessor。
//
#pragma once

#include <string>
#include <vector>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/insightface_types.h"
#include "vision/face/insightface/scrfd/insightface_scrfd_preprocessor.h"
#include "vision/face/insightface/scrfd/insightface_scrfd_postprocessor.h"

namespace modeldeploy::vision::face {

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
