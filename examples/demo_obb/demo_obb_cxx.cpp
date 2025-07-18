//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    // option.enable_trt = true;

    modeldeploy::vision::detection::UltralyticsObb yolov8("../../test_data/test_models/yolo11n-obb_nms.onnx", option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_obb1.jpg");
    std::vector<modeldeploy::vision::ObbResult> result;
    TimerArray timers;
    for (int i = 0; i < 100; i++) {
        yolov8.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    const auto vis_image =
        modeldeploy::vision::vis_obb(img, result, 0.2, "../../test_data/test_models/msyh.ttc", 12, 0.3, 0);
    vis_image.imshow("test");
}
