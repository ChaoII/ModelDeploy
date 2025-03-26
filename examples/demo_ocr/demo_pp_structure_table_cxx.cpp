//
// Created by aichao on 2025/3/21.
//

#include <uni_algo.h>

#include "csrc/vision.h"
#include "csrc/vision/common/result.h"
#ifdef _WIN32
#include "windows.h"
#endif

cv::Mat VisOcr(const cv::Mat& im, const modeldeploy::vision::OCRResult& ocr_result,
               const float score_threshold) {
    auto vis_im = im.clone();
    bool have_score =
      (ocr_result.boxes.size() == ocr_result.rec_scores.size());

    for (int n = 0; n < ocr_result.boxes.size(); n++) {
        if (have_score) {
            if (ocr_result.rec_scores[n] < score_threshold) {
                continue;
            }
        }
        cv::Point rook_points[4];

        for (int m = 0; m < 4; m++) {
            rook_points[m] = cv::Point(int(ocr_result.boxes[n][m * 2]),
                                       int(ocr_result.boxes[n][m * 2 + 1]));
        }

        const cv::Point* ppt[1] = {rook_points};
        int npt[] = {4};
        cv::polylines(vis_im, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    return vis_im;
}

int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    // The rec model can inference a batch of images now.
    // User could initialize the inference batch size and set them after create
    // PP-OCR model.
    const std::string& det_model_file = "../../test_data/test_models/ocr/repsvtr_mobile/det_infer.onnx";
    const std::string& rec_model_file = "../../test_data/test_models/ocr/repsvtr_mobile/rec_infer.onnx";
    const std::string& table_model_file = "../../test_data/test_models/ocr/SLANet_plus.onnx";
    const std::string& rec_label_file = "../../test_data/key.txt";
    const std::string& table_char_dict_path = "../../test_data/table_structure_dict_ch.txt";
    const std::string& image_file = "../../test_data/test_images/test_table.jpg";


    int rec_batch_size = 1;

    auto det_model = modeldeploy::vision::ocr::DBDetector(det_model_file);
    auto rec_model = modeldeploy::vision::ocr::Recognizer(
        rec_model_file, rec_label_file);
    auto table_model = modeldeploy::vision::ocr::StructureV2Table(
        table_model_file, table_char_dict_path);

    assert(det_model.Initialized());
    assert(rec_model.Initialized());
    assert(table_model.Initialized());

    // Parameters settings for pre and post processing of Det/Cls/Rec Models.
    // All parameters are set to default values.
    det_model.get_preprocessor().set_max_side_len(960);
    det_model.get_postprocessor().set_det_db_thresh(0.3);
    det_model.get_postprocessor().set_det_db_box_thresh(0.6);
    det_model.get_postprocessor().set_det_db_unclip_ratio(1.5);
    det_model.get_postprocessor().set_det_db_score_mode("slow");
    det_model.get_postprocessor().set_use_dilation(0);

    rec_model.get_preprocessor().set_rec_image_shape({3, 48, 320});

    // The classification model is optional, so the PP-OCR can also be connected
    // in series as follows
    auto ppstructurev2_table = modeldeploy::pipeline::PPStructureV2Table(
        &det_model, &rec_model, &table_model);

    // Set inference batch size for cls model and rec model, the value could be -1
    // and 1 to positive infinity.
    // When inference batch size is set to -1, it means that the inference batch
    // size of the rec models will be the same as the number of boxes detected
    // by the det model.
    ppstructurev2_table.set_rec_batch_size(rec_batch_size);

    if (!ppstructurev2_table.initialized()) {
        std::cerr << "Failed to initialize PP-OCR-Table." << std::endl;
        return -1;
    }

    auto im = cv::imread(image_file);
    auto im_bak = im.clone();

    modeldeploy::vision::OCRResult result;
    if (!ppstructurev2_table.predict(&im, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return -1;
    }
    std::cout << result.Str() << std::endl;
}
