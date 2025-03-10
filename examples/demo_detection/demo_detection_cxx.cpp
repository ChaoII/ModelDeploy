//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"

int main() {
     modeldeploy::vision::detection::YOLOv8 yolov8("../tests/test_models/best.onnx");
     bool is_inited = yolov8.initialized();
     auto img = cv::imread("../tests/test_images/test_detection.png");
     modeldeploy::vision::DetectionResult result;
     yolov8.get_preprocessor().set_size({1440,1440});
     yolov8.predict(img, &result);
     std::cout << result.Str() << std::endl;
}
