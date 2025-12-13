//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    modeldeploy::vision::classification::Classification cls_model("../../test_data/test_models/zhgd_ml.onnx",
                                                                  option);
    cls_model.get_preprocessor().set_size({192, 256});
    cls_model.get_preprocessor().disable_center_crop();
    cls_model.get_postprocessor().set_multi_label(true);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_face.jpg");
    modeldeploy::vision::ClassifyResult results;
    cls_model.predict(img, &results);
    const auto vis_img = modeldeploy::vision::vis_cls(img, results, 1, 0.5,
                                                      "../../test_data/test_models/msyh.ttf",
                                                      12, 0.3, false);
    img.imshow("cls");
    modeldeploy::vision::dis_cls(results);
}
