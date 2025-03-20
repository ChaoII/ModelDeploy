//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"

int main() {
     modeldeploy::vision::detection::YOLOv8 yolov8("../../test_data/test_models/best.onnx");
     const auto img = cv::imread("../../test_data/test_images/test_detection.png");
     modeldeploy::vision::DetectionResult result;
     yolov8.get_preprocessor().set_size({1440,1440});
     yolov8.predict(img, &result);
     std::cout << result.Str() << std::endl;
}
