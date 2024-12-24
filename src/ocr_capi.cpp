#include "ocr_capi.h"
#include "utils.h"
#include "fastdeploy/vision.h"

using DBDetector = fastdeploy::vision::ocr::DBDetector;
using Recognizer = fastdeploy::vision::ocr::Recognizer;
using PPOCRv4 = fastdeploy::pipeline::PPOCRv4;


WRect get_template_position(WImage *shot_img, WImage *template_img) {
    cv::Mat shot_mat = wimage_to_mat(shot_img);
    cv::Mat template_mat = wimage_to_mat(template_img);
    int h = template_mat.rows;
    int w = template_mat.cols;
    // 准备输出结果
    cv::Mat result;
    // 模板匹配
    cv::matchTemplate(shot_mat, template_mat, result, cv::TM_SQDIFF_NORMED);
    // 寻找最佳匹配位置
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
    // 绘制矩形框
    return WRect{min_loc.x, min_loc.y, w, h};
}

WModel *create_ocr_model(OCRModelParameters *parameters) {
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
    assert(det_model.Initialized());
    assert(rec_model.Initialized());

    det_model->GetPreprocessor().SetMaxSideLen(parameters->max_side_len);
    det_model->GetPostprocessor().SetDetDBThresh(parameters->det_db_thresh);
    det_model->GetPostprocessor().SetDetDBBoxThresh(parameters->det_db_box_thresh);
    det_model->GetPostprocessor().SetDetDBUnclipRatio(parameters->det_db_unclip_ratio);
    det_model->GetPostprocessor().SetDetDBScoreMode(parameters->det_db_score_mode);
    det_model->GetPostprocessor().SetUseDilation(parameters->use_dilation);
    auto model = (WModel *) malloc(sizeof(WModel));
    auto ocr_model = new fastdeploy::pipeline::PPOCRv4(det_model, rec_model);
    auto model_name = ocr_model->ModelName();
    model->format = parameters->format;
    model->model_content = ocr_model;
    model->model_name = (char *) malloc((ocr_model->ModelName().size() + 1) * sizeof(char));
    model_name.copy(model->model_name, model_name.size());
    ocr_model->SetRecBatchSize(parameters->rec_batch_size);
    if (!(ocr_model->Initialized())) {
        std::cerr << "Failed to initialize PP-OCR." << std::endl;
        return nullptr;
    }
    return model;
}

WRect get_text_position(OCRModelHandle model, WImage *image, const char *text) {
    cv::Mat cv_image = wimage_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model);
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


StatusCode text_rec_buffer(OCRModelHandle model, WImage *image, WOCRResult *out_data, int draw_result,
                           WColor color, double alpha, int is_save_result) {

    auto cv_image = wimage_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model);
    bool res_status = ocr_model->Predict(cv_image, &res);
    if (!res_status) {
        return StatusCode::ModelPredictFailed;
    }
    auto r_size = res.boxes.size();
    out_data->size = r_size;
    out_data->boxes = (WPolygon *) calloc(r_size, sizeof(WPolygon));
    out_data->scores = (float *) calloc(r_size, sizeof(float));
    out_data->texts = (char **) calloc(r_size, sizeof(char *));
    for (int i = 0; i < r_size; ++i) {
        auto text = res.text[i];
        out_data->texts[i] = (char *) calloc(text.size() + 1, sizeof(char));
        strcpy_s(out_data->texts[i], text.size() + 1, text.c_str());
        WPolygon polygon{};
        for (int j = 0; j < 4; ++j) {
            WPoint point;
            point.x = res.boxes[i][j * 2];
            point.y = res.boxes[i][j * 2 + 1];
            polygon.points[j] = point;
        }
        out_data->boxes[i] = polygon;
        out_data->scores[i] = res.rec_scores[i];
        if (draw_result) {
            auto p1 = cv::Point(res.boxes[i][0], res.boxes[i][1]);
            auto p2 = cv::Point(res.boxes[i][2], res.boxes[i][3]);
            auto p3 = cv::Point(res.boxes[i][4], res.boxes[i][5]);
            auto p4 = cv::Point(res.boxes[i][6], res.boxes[i][7]);
            draw_transparent_rectangle(cv_image, {p1, p2, p3, p4},
                                       cv::Scalar(color.b, color.g, color.r), alpha);
            cv::FontFace font("msyh.ttc");
            cv::putText(cv_image, res.text[i], cv::Point(p1.x, p1.y - 3),
                        cv::Scalar(color.b, color.g, color.r), font, 20);
        }
    }
    if (is_save_result > 0) {
        cv::imwrite("vis_result.jpg", cv_image);
    }
    return StatusCode::Success;
}

void print_ocr_result(WOCRResult *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_polygon(result->boxes[i]) << "text: "
                  << result->texts[i] << " score: " << result->scores[i] << std::endl;
    }
}

void free_ocr_result(WOCRResult *result) {
    if (!result) {
        return;
    }
    free(result->boxes);
    free(result->scores);
    for (int i = 0; i < result->size; ++i) {
        free(result->texts[i]);
    }
    free(result->texts);
    result->size = 0;
}

void free_ocr_model(OCRModelHandle model) {
    delete static_cast<PPOCRv4 *>(model);
}






