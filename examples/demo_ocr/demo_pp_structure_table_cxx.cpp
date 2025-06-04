//
// Created by aichao on 2025/3/21.
//

#include "csrc/vision.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"
#ifdef _WIN32
#include "windows.h"
#endif

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    // The rec model can inference a batch of images now.
    // User could initialize the inference batch size and set them after create
    // PP-OCR model.
    const std::string& det_model_file = "../../test_data/test_models/ocr/repsvtr_mobile/det_infer.onnx";
    const std::string& rec_model_file = "../../test_data/test_models/ocr/repsvtr_mobile/rec_infer.onnx";
    const std::string& table_model_file = "../../test_data/test_models/ocr/SLANeXt_wired.onnx";
    const std::string& rec_label_file = "../../test_data/ppocrv4_dict.txt";
    const std::string& table_char_dict_path = "../../test_data/table_structure_dict_ch.txt";
    const std::string& image_file = "../../test_data/test_images/test_table.jpg";

    constexpr int rec_batch_size = 8;


    // The classification model is optional, so the PP-OCR can also be connected
    // in series as follows
    auto pp_structure_v2_table = modeldeploy::vision::ocr::PPStructureV2Table(
        det_model_file, rec_model_file, table_model_file, rec_label_file, table_char_dict_path
    );
    // Set inference batch size for cls model and rec model, the value could be -1
    // and 1 to positive infinity.
    // When inference batch size is set to -1, it means that the inference batch
    // size of the rec models will be the same as the number of boxes detected
    // by the det model.
    pp_structure_v2_table.set_rec_batch_size(rec_batch_size);
    auto im = cv::imread(image_file);
    auto im_bak = im.clone();
    modeldeploy::vision::OCRResult result;
    if (!pp_structure_v2_table.predict(&im, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    auto vis_image = modeldeploy::vision::vis_ocr(im_bak, result, "../../test_data/msyh.ttc", 20, 0.5, 0);

    cv::imshow("result", vis_image);
    cv::waitKey(0);

    modeldeploy::vision::dis_ocr(result);
}
