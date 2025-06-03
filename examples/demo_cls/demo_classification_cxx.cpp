//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    modeldeploy::vision::classification::UltralyticsCls yol_ov5_cls("../../test_data/test_models/yolo11n-cls.onnx",
                                                                    option);
    auto img = cv::imread("../../test_data/test_images/test_face.jpg");
    modeldeploy::vision::ClassifyResult results;
    yol_ov5_cls.predict(img, &results);
    const auto vis_img = modeldeploy::vision::vis_classification(img, results, 1, 0.5,
                                                                 "../../test_data/test_models/msyh.ttf",
                                                                 12, 0.3, false);
    cv::imshow("test", vis_img);
    cv::waitKey(0);
    results.display();
}
