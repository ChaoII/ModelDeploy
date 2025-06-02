//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::vision::detection::UltralyticsPose yolov8("../../test_data/test_models/yolo11n-pose.onnx");
    auto img = cv::imread("../../test_data/test_images/test_pose0.jpg");
    modeldeploy::vision::PoseResult result;
    yolov8.get_preprocessor().set_size({640, 640});
    yolov8.get_preprocessor().set_mini_pad(false);
    yolov8.predict(img, &result);
    result.display();
    const auto vis_image =
        modeldeploy::vision::vis_pose(img, result, "../../test_data/test_models/msyh.ttc", 12, 4, 0.3, false);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
