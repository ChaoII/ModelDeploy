//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::vision::detection::UltralyticsObb yolov8("../../test_data/test_models/yolov8l-obb.onnx");
    auto img = cv::imread("../../test_data/test_images/test_obb2.jpg");
    modeldeploy::vision::DetectionResult result;
    yolov8.get_preprocessor().set_size({1024, 1024});
    yolov8.predict(img, &result);
    result.display();
    const auto vis_image =
        modeldeploy::vision::vis_detection(img, result, 0.2, "../../test_data/test_models/msyh.ttc", 12, 0.3, 0);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
