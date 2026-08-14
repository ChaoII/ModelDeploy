//
// insightface buffalo_l det_10g：SCRFD 风格人脸检测。
// 输出 9 个 tensor（3 stride x score/bbox/kps），与 python insightface SCRFD 对齐。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/insightface_types.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceDet : public BaseModel {
    public:
        explicit InsightFaceDet(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceDet"; }

        // 检测：输入 BGR 图，输出人脸框（原图坐标）+ 5 关键点（原图坐标）
        bool predict(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                     TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceDet> clone() const;

        // 检测阈值 / NMS 阈值（对齐 python 默认）
        float det_thresh_ = 0.5f;
        float nms_thresh_ = 0.4f;
        // 输入尺寸（默认 640x640）
        std::vector<int> input_size_{640, 640};

    protected:
        bool Initialize();
    };

} // namespace modeldeploy::vision::face
