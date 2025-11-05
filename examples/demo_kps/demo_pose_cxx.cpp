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
    modeldeploy::vision::detection::UltralyticsPose yolov8("../../test_data/test_models/zc.onnx", option);
    yolov8.get_postprocessor().set_keypoints_num(2);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/zc0.jpg");
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
        modeldeploy::vision::vis_keypoints(img, result, "../../test_data/test_models/msyh.ttc", 12, 4, 0.3, false);
    vis_image.imshow("result");
}
