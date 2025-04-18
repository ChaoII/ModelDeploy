//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::vision::detection::YOLOv8 yolov8("../../test_data/test_models/best.onnx");
    auto img = cv::imread("../../test_data/test_images/test_detection.png");
    modeldeploy::vision::DetectionResult result;
    yolov8.get_preprocessor().set_size({1440, 1440});
    yolov8.predict(img, &result);
    std::cout<<"-------------"<<std::endl;
    result.display();
    std::cout<<"-------------"<<std::endl;

    const auto vis_image =
        modeldeploy::vision::vis_detection(img, result, "../../test_data/test_models/font.ttf", 12, 0.3, 0);
    std::cout<<"-------------"<<std::endl;

    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
