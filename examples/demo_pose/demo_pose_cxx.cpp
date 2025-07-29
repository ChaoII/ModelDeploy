//
// Created by aichao on 2025/2/24.
//

#include <chrono>
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_trt = true;
    option.enable_fp16 = true;
    modeldeploy::vision::detection::UltralyticsPose yolov8("../../test_data/test_models/yolo11n-pose_nms.onnx", option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_person.jpg");
    std::vector<modeldeploy::vision::PoseResult> result;
    int warm_up_count = 20;
    for (int i = 0; i < warm_up_count; ++i) {
        yolov8.predict(img, &result);
    }

    TimerArray timers;
    int loop_count = 80;
    for (int i = 0; i < loop_count; ++i) {
        yolov8.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    // result.display();
    const auto vis_image =
        modeldeploy::vision::vis_pose(img, result, "../../test_data/test_models/msyh.ttc", 12, 4, 0.3, false);
    vis_image.imshow("result");
}
