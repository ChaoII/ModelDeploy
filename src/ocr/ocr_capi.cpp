#include "ocr_capi.h"
#include "../utils/internal/utils.h"
#include "fastdeploy/vision.h"


using DBDetector = fastdeploy::vision::ocr::DBDetector;
using Recognizer = fastdeploy::vision::ocr::Recognizer;
using PPOCRv4 = fastdeploy::pipeline::PPOCRv4;




MDStatusCode md_create_ocr_model(MDModel *model, MDOCRModelParameters *parameters) {
    if (!model) {
        return MDStatusCode::MemoryAllocatedFailed;
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
        return MDStatusCode::OCRDetModelInitializeFailed;
    }
    if (!rec_model->Initialized()) {
        std::cerr << "Failed to initialize OCR recognition model." << std::endl;
        return MDStatusCode::OCRRecModelInitializeFailed;
    }

    det_model->GetPreprocessor().SetMaxSideLen(parameters->max_side_len);
    det_model->GetPostprocessor().SetDetDBThresh(parameters->det_db_thresh);
    det_model->GetPostprocessor().SetDetDBBoxThresh(parameters->det_db_box_thresh);
    det_model->GetPostprocessor().SetDetDBUnclipRatio(parameters->det_db_unclip_ratio);
    det_model->GetPostprocessor().SetDetDBScoreMode(parameters->det_db_score_mode);
    det_model->GetPostprocessor().SetUseDilation(parameters->use_dilation);

    auto ocr_model = new fastdeploy::pipeline::PPOCRv4(det_model, rec_model);
    model->type = MDModelType::OCR;
    model->format = parameters->format;
    model->model_content = ocr_model;
    model->model_name = strdup(ocr_model->ModelName().c_str());
    ocr_model->SetRecBatchSize(parameters->rec_batch_size);
    if (!(ocr_model->Initialized())) {
        std::cerr << "Failed to initialize OCR model." << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDRect md_get_text_position(MDModel *model, MDImage *image, const char *text) {
    cv::Mat cv_image = md_image_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model->model_content);
    bool res_status = ocr_model->Predict(cv_image, &res);
    if (!res_status) {
        return MDRect{0, 0, 0, 0};
    }
    for (int i = 0; i < res.boxes.size(); ++i) {
        std::vector<cv::Point> polygon;
        if (contains_substring(res.text[i], text)) {
            for (int j = 0; j < 4; ++j) {
                polygon.emplace_back(res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]);
            }
            cv::Rect boundingRect = cv::boundingRect(polygon);
            return MDRect{boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height};
        }
    }
    return MDRect{0, 0, 0, 0};
}


MDStatusCode md_ocr_model_predict(MDModel *model, MDImage *image, MDOCRResults *results) {
    auto cv_image = md_image_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model->model_content);
    bool res_status = ocr_model->Predict(cv_image, &res);
    if (!res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    auto r_size = res.boxes.size();
    results->size = (int)r_size;
    if (r_size == 0) {
        results->data = nullptr;
        return MDStatusCode::Success;
    }
    results->data = (MDOCRResult *) malloc(sizeof(MDOCRResult) * r_size);
    for (int i = 0; i < r_size; ++i) {
        auto text = res.text[i];
        results->data[i].text = (char *) malloc(text.size() + 1);
        memcpy(results->data[i].text, text.c_str(), text.size() + 1);
        results->data[i].score = res.rec_scores[i];
        MDPolygon polygon{(MDPoint *) malloc(sizeof(MDPoint) * 4), 4};
        for (int j = 0; j < 4; ++j) {
            polygon.data[j] = {res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]};
        }
        results->data[i].box = polygon;
    }
    return MDStatusCode::Success;
}

void md_print_ocr_result(MDOCRResults *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_polygon(result->data[i].box) << " text: "
                  << result->data[i].text << " score: " << result->data[i].score << std::endl;
    }
}

void md_draw_ocr_result(MDImage *image, MDOCRResults *results, const char *font_path, int font_size, MDColor *color,
                        double alpha, int save_result) {
    cv::Mat cv_image, overlay;
    cv_image = md_image_to_mat(image);
    cv_image.copyTo(overlay);
    cv::FontFace font(font_path);
    cv::Scalar cv_color(color->b, color->g, color->r);
    // 绘制半透明部分（填充矩形）
    for (int i = 0; i < results->size; ++i) {
        auto polygon = results->data[i].box;
        std::vector<cv::Point> points;
        points.reserve(polygon.size);
        for (int j = 0; j < polygon.size; ++j) {
            points.emplace_back(polygon.data[j].x, polygon.data[j].y);
        }
        cv::fillPoly(overlay, points, cv_color, cv::LINE_AA, 0);
        auto size = cv::getTextSize(cv::Size(0, 0), results->data[i].text,
                                    {points[0].x, points[0].y}, font, font_size);
        cv::rectangle(cv_image, size, cv_color, -1, cv::LINE_AA, 0);

    }
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    // 绘制非半透明部分（矩形边框、文字等）
    for (int i = 0; i < results->size; ++i) {
        auto polygon = results->data[i].box;
        std::vector<cv::Point> points;
        points.reserve(polygon.size);
        for (int j = 0; j < polygon.size; ++j) {
            points.emplace_back(polygon.data[j].x, polygon.data[j].y);
        }
        cv::polylines(cv_image, points, true, cv_color, 1, cv::LINE_AA, 0);
        auto size = cv::getTextSize(cv::Size(0, 0), results->data[i].text,
                                    {points[0].x, points[0].y}, font, font_size);
        cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
        cv::putText(cv_image, results->data[i].text, {points[0].x, points[0].y - 2},
                    cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
    }

    if (save_result) {
        cv::imwrite("vis_result.jpg", cv_image);
    }

}

void md_free_ocr_result(MDOCRResults *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        free(result->data[i].text);
        free(result->data[i].box.data);
    }
    free(result->data);
    result->data = nullptr;
    result->size = 0;
}

void md_free_ocr_model(MDModel *model) {
    if (model != nullptr) {
        if (model->model_content != nullptr) {
            delete static_cast<PPOCRv4 *>(model->model_content);
            model->model_content = nullptr;
        }
        free(model->model_name);
        model->model_name = nullptr;
    }
}






