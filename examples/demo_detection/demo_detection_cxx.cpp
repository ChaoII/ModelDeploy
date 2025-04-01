//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::vision::detection::YOLOv8 yolov8("../../test_data/test_models/best.onnx");
    const auto img = cv::imread("../../test_data/test_images/test_detection.png");
    modeldeploy::vision::DetectionResult result;
    yolov8.get_preprocessor().set_size({1440, 1440});
    yolov8.predict(img, &result);
    result.display();
    auto vis_image = modeldeploy::vision::VisDetection(img, result, "../../test_data/test_models/font.ttf", 20, 0.5, 0);

    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
