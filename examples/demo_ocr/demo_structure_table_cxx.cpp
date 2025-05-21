//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision.h"
#include "csrc/vision/common/result.h"

int main() {
    auto table_model = modeldeploy::vision::ocr::StructureV2Table(
        "../../test_data/test_models/ocr/SLANeXt_wired.onnx",
        "../../test_data/table_structure_dict_ch.txt");
    assert(table_model.is_initialized());
    auto im = cv::imread("../../test_data/test_images/test_table1.jpg");
    auto im_bak = im.clone();
    modeldeploy::vision::OCRResult result;
    if (!table_model.predict(im, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << result.str() << std::endl;
    return 0;
}
