//
// Created by aichao on 2025/2/24.
//

#include <chrono>
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    modeldeploy::vision::detection::UltralyticsPose yolov8("../../test_data/test_models/onnx/yolo11n/yolo11n-pose.onnx", option);
    yolov8.get_postprocessor().set_keypoints_num(17);
    auto img = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_person.jpg");
    std::vector<modeldeploy::vision::KeyPointsResult> result;
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
        modeldeploy::vision::vis_pose(img, result, "../../test_data/msyh.ttc", 12, 4, 0.3, false);
    (void)vis_image.imwrite("pose_out.jpg");
}
