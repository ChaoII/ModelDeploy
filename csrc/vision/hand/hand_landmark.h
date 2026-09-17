//
// MediaPipe 手部关键点（21 点）独立模型封装。
// 与 hand::HandKeypoint（YOLO-Pose 手部模型薄封装）不同：本类对应 OpenCV Zoo 的
// handpose_estimation_mediapipe_2023feb.onnx —— 单一 landmark 模型，输入 224x224 NHWC，
// 输出 4 个张量：landmarks(1x63) / handflag(1x1) / handedness(1x1) / world_landmarks(1x63)。
// 官方 pipeline 需先由 palm detector 裁剪/旋转出「手掌区域」再送入本模型；本类只负责模型本身，
// 是否裁剪由调用方决定（无 palm detector 时调用方可整帧/ROI 直接送入，见批次报告的限制说明）。
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"

namespace modeldeploy::vision::hand {
    /*! @brief MediaPipe 手部 21 点关键点模型。
     *  \param[in] model_file handpose_estimation_mediapipe ONNX 路径
     *  predict 对传入图像估计至多一个手部 KeyPointsResult：
     *  - keypoints：21 个 Point3f，x/y 为**传入图像像素坐标**（由模型 224 输入空间线性回映射），
     *    z 为 MediaPipe 相对深度（原样保留，不参与事件契约的 score 位）；
     *  - box：21 点轴对齐外接框（像素，已钳制到图像内，宽高至少 1）；
     *  - score：手部置信度（Identity_1 的 sigmoid 输出，即 handflag）；
     *  - label_id：0。
     */
    class MODELDEPLOY_CXX_EXPORT HandLandmark : public BaseModel {
    public:
        explicit HandLandmark(const std::string& model_file,
                              const RuntimeOption& option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "HandLandmark"; }

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr);

        [[nodiscard]] std::unique_ptr<HandLandmark> clone() const;

    private:
        bool initialize();
    };
} // namespace modeldeploy::vision::hand
