#include "ocr_capi.h"
#include "utils.h"
#include "fastdeploy/vision.h"

using DBDetector = fastdeploy::vision::ocr::DBDetector;
using Recognizer = fastdeploy::vision::ocr::Recognizer;
using PPOCRv4 = fastdeploy::pipeline::PPOCRv4;


StatusCode create_ocr_model(WModel *model, OCRModelParameters *parameters) {
    if (!model) {
        return StatusCode::MemoryAllocatedFailed;
    }
    fastdeploy::SetLogger(false);
    fastdeploy::RuntimeOption option;
    option.UseOrtBackend();
    option.SetCpuThreadNum(parameters->thread_num);

    auto det_model_file = std::string(parameters->model_dir) + "/det/" + "inference.pdmodel";
    auto det_params_file = std::string(parameters->model_dir) + "/det/" + "inference.pdiparams";
    auto rec_model_file = std::string(parameters->model_dir) + "/rec/" + "inference.pdmodel";
    auto rec_params_file = std::string(parameters->model_dir) + "/rec/" + "inference.pdiparams";

    auto det_model = new DBDetector(det_model_file, det_params_file, option);
    auto rec_model = new Recognizer(rec_model_file, rec_params_file, parameters->dict_path, option);
    if (!det_model->Initialized()) {
        std::cerr << "Failed to initialize OCR detection model." << std::endl;
        return StatusCode::OCRDetModelInitializeFailed;
    }
    if (!rec_model->Initialized()) {
        std::cerr << "Failed to initialize OCR recognition model." << std::endl;
        return StatusCode::OCRRecModelInitializeFailed;
    }

    det_model->GetPreprocessor().SetMaxSideLen(parameters->max_side_len);
    det_model->GetPostprocessor().SetDetDBThresh(parameters->det_db_thresh);
    det_model->GetPostprocessor().SetDetDBBoxThresh(parameters->det_db_box_thresh);
    det_model->GetPostprocessor().SetDetDBUnclipRatio(parameters->det_db_unclip_ratio);
    det_model->GetPostprocessor().SetDetDBScoreMode(parameters->det_db_score_mode);
    det_model->GetPostprocessor().SetUseDilation(parameters->use_dilation);

    auto ocr_model = new fastdeploy::pipeline::PPOCRv4(det_model, rec_model);
    auto model_name = ocr_model->ModelName();
    model->type = ModelType::OCR;
    model->format = parameters->format;
    model->model_content = ocr_model;
    model->model_name = (char *) malloc((ocr_model->ModelName().size() + 1) * sizeof(char));
    memcpy(model->model_name, model_name.c_str(), model_name.size() + 1);
    ocr_model->SetRecBatchSize(parameters->rec_batch_size);
    if (!(ocr_model->Initialized())) {
        std::cerr << "Failed to initialize OCR model." << std::endl;
        return StatusCode::ModelInitializeFailed;
    }
    return StatusCode::Success;
}

WRect get_text_position(WModel *model, WImage *image, const char *text) {
    cv::Mat cv_image = wimage_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model->model_content);
    bool res_status = ocr_model->Predict(cv_image, &res);
    if (!res_status) {
        return WRect{0, 0, 0, 0};
    }
    for (int i = 0; i < res.boxes.size(); ++i) {
        std::vector<cv::Point> polygon;
        if (contains_substring(res.text[i], text)) {
            for (int j = 0; j < 4; ++j) {
                polygon.emplace_back(res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]);
            }
            cv::Rect boundingRect = cv::boundingRect(polygon);
            return WRect{boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height};
        }
    }
    return WRect{0, 0, 0, 0};
}


StatusCode ocr_model_predict(WModel *model, WImage *image, WOCRResults *results, int draw_result,
                             WColor color, double alpha, int is_save_result) {
    cv::FontFace font("msyh.ttc");
    auto cv_image = wimage_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model->model_content);
    bool res_status = ocr_model->Predict(cv_image, &res);
    if (!res_status) {
        return StatusCode::ModelPredictFailed;
    }
    auto r_size = res.boxes.size();
    results->size = r_size;
    if (r_size == 0) {
        results->data = nullptr;
        return StatusCode::Success;
    }
    results->data = (WOCRResult *) malloc(sizeof(WOCRResult) * r_size);
    for (int i = 0; i < r_size; ++i) {
        auto text = res.text[i];
        results->data[i].text = (char *) malloc(text.size() + 1);
        memcpy(results->data[i].text, text.c_str(), text.size() + 1);
        results->data[i].score = res.rec_scores[i];
        WPolygon polygon{(WPoint *) malloc(sizeof(WPoint) * 4), 4};
        for (int j = 0; j < 4; ++j) {
            polygon.data[j] = {res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]};
        }
        results->data[i].box = polygon;
        if (draw_result) {
            auto p1 = cv::Point(res.boxes[i][0], res.boxes[i][1]);
            auto p2 = cv::Point(res.boxes[i][2], res.boxes[i][3]);
            auto p3 = cv::Point(res.boxes[i][4], res.boxes[i][5]);
            auto p4 = cv::Point(res.boxes[i][6], res.boxes[i][7]);
            draw_transparent_rectangle(cv_image, {p1, p2, p3, p4},
                                       cv::Scalar(color.b, color.g, color.r), alpha);
            cv::putText(cv_image, res.text[i], cv::Point(p1.x, p1.y - 3),
                        cv::Scalar(color.b, color.g, color.r), font, 20);
        }
    }
    if (is_save_result > 0) {
        cv::imwrite("vis_result.jpg", cv_image);
    }
    return StatusCode::Success;
}

void print_ocr_result(WOCRResults *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_polygon(result->data[i].box) << " text: "
                  << result->data[i].text << " score: " << result->data[i].score << std::endl;
    }
}

void free_ocr_result(WOCRResults *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        free(result->data[i].text);
        free(result->data[i].box.data);
    }
    free(result->data);
    result->size = 0;
}

void free_ocr_model(WModel *model) {
    if (!model) return;
    free(model->model_name);
    delete static_cast<PPOCRv4 *>(model->model_content);
}






