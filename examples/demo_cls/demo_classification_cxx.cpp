//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    modeldeploy::vision::classification::UltralyticsCls yol_ov5_cls("../../test_data/test_models/yolo11n-cls.onnx",
                                                                    option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_face.jpg");
    modeldeploy::vision::ClassifyResult results;
    yol_ov5_cls.predict(img, &results);
    const auto vis_img = modeldeploy::vision::vis_cls(img, results, 1, 0.5,
                                                      "../../test_data/test_models/msyh.ttf",
                                                      12, 0.3, false);
    img.imshow("cls");
    modeldeploy::vision::dis_cls(results);
}
