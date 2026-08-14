//
// insightface buffalo_l landmark：2d106det（2D 106 点）+ 1k3d68（3D 68 点 + 姿态）。
// 与 python insightface.model_zoo.landmark.Landmark 逐值对齐。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/insightface_types.h"
#include "vision/face/insightface/face_align_utils.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmark : public BaseModel {
    public:
        explicit InsightFaceLandmark(const std::string& model_file,
                                     const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceLandmark"; }

        // 计算 2D 关键点：输入 BGR 原图 + 人脸框，输出 106 个点（原图坐标）
        bool predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                           std::vector<std::array<float, 2>>* landmarks,
                           TimerArray* timers = nullptr);

        // 计算 3D 关键点：输入 BGR 原图 + 人脸框，输出 68 个 3D 点 + 姿态
        bool predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                          std::vector<std::array<float, 3>>* landmarks,
                          std::array<float, 3>* pose,
                          TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceLandmark> clone() const;

        // 输入尺寸（2d106 和 1k3d68 都是 192）
        std::vector<int> input_size_{192, 192};

    protected:
        bool Initialize();

    private:
        // 以 bbox 中心 transform 到 input_size，返回裁剪图 + 仿射矩阵
        bool crop_face(const ImageData& image, const std::array<float, 4>& bbox,
                       cv::Mat* aimg, cv::Mat* M);
    };

} // namespace modeldeploy::vision::face
