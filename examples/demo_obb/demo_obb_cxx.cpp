//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();

    modeldeploy::vision::detection::UltralyticsObb yolov8("../../test_data/test_models/onnx/yolo11n/yolo11n-obb_nms.onnx", option);
    auto img = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_obb1.jpg");
    std::vector<modeldeploy::vision::ObbResult> result;
    TimerArray timers;
    for (int i = 0; i < 100; i++) {
        yolov8.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    const auto vis_image =
        modeldeploy::vision::vis_obb(img, result, 0.2, "../../test_data/msyh.ttc", 12, 0.3, 0);
    (void)vis_image.imwrite("obb_out.jpg");
}
