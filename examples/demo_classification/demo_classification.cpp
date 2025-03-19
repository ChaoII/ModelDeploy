//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"

int main() {
    modeldeploy::vision::classification::YOLOv5Cls yol_ov5_cls("../test_data/test_models/yolov5n-cls.onnx");
    auto img = cv::imread("../test_data/test_images/test_face.jpg");
    modeldeploy::vision::ClassifyResult results;
    yol_ov5_cls.Predict(img, &results);
    std::cout << results.Str() << std::endl;
}
