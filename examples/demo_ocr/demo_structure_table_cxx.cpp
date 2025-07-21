//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    auto table_model = modeldeploy::vision::ocr::StructureV2Table(
        "../../test_data/test_models/ocr/SLANet_plus.onnx",
        "../../test_data/table_structure_dict_ch.txt");
    auto im = modeldeploy::ImageData::imread("../../test_data/test_images/test_table1.jpg");
    auto im_bak = im.clone();
    modeldeploy::vision::OCRResult result;
    if (!table_model.predict(im, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    auto vis_image = modeldeploy::vision::vis_ocr(im_bak, result, "../../test_data/msyh.ttc", 20, 0.5, 0);
    vis_image.imshow("result");
    modeldeploy::vision::dis_ocr(result);
    return 0;
}
