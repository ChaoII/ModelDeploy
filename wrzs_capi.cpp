#include "wrzs_capi.h"
#include "utils.h"
#include "fastdeploy/vision.h"

using DBDetector = fastdeploy::vision::ocr::DBDetector;
using Recognizer = fastdeploy::vision::ocr::Recognizer;
using PPOCRv4 = fastdeploy::pipeline::PPOCRv4;

WPoint get_center_point(WRect rect) {
    return WPoint{rect.x + rect.width / 2, rect.y + rect.height / 2};
}

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

OCRModelHandle create_ocr_model(const char *model_dir, const char *dict_file, int thread_num) {
    fastdeploy::SetLogger(false);
    fastdeploy::RuntimeOption option;
    option.UseOrtBackend();
    option.SetCpuThreadNum(thread_num);

    auto det_model_file = std::string(model_dir) + "/det/" + "inference.pdmodel";
    auto det_params_file = std::string(model_dir) + "/det/" + "inference.pdiparams";
    auto rec_model_file = std::string(model_dir) + "/rec/" + "inference.pdmodel";
    auto rec_params_file = std::string(model_dir) + "/rec/" + "inference.pdiparams";

    auto det_model = new DBDetector(det_model_file, det_params_file, option);
    auto rec_model = new Recognizer(rec_model_file, rec_params_file, dict_file, option);
    assert(det_model.Initialized());
    assert(rec_model.Initialized());

    det_model->GetPreprocessor().SetMaxSideLen(960);
//    det_model->GetPostprocessor().SetDetDBThresh(0.3);
    det_model->GetPostprocessor().SetDetDBBoxThresh(0.6);
    det_model->GetPostprocessor().SetDetDBUnclipRatio(1.5);
    det_model->GetPostprocessor().SetDetDBScoreMode("slow");
//    det_model.GetPostprocessor().SetUseDilation(0);
    auto ocr_model = new fastdeploy::pipeline::PPOCRv4(det_model, rec_model);
    ocr_model->SetRecBatchSize(6);
    if (!(ocr_model->Initialized())) {
        std::cerr << "Failed to initialize PP-OCR." << std::endl;
        return nullptr;
    }
    return ocr_model;
}

WRect get_text_position(OCRModelHandle model, WImage *image, const char *text) {
    cv::Mat cv_image = wimage_to_mat(image);
    fastdeploy::vision::OCRResult res;
    auto ocr_model = static_cast<PPOCRv4 *> (model);
    bool res_status = ocr_model->Predict(cv_image, &res);
    std::cout << res.Str() << std::endl;
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
            std::cout << boundingRect << std::endl;
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

void free_wimage(WImage *img) {
    img->width = 0;
    img->height = 0;
    img->type = WImageType::WImageType_BGR;
    free(img->data);
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
    model = nullptr;
}

WImage *read_image(const char *path) {
    cv::Mat image = cv::imread(path);
    if (!image.empty()) {
        return mat_to_wimage(image);
    }
    return nullptr;
}


