//
// Created by AC on 2024-12-24.
//

#include <fstream>
#include <random>
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "csrc/vision/utils.h"

#include <csrc/core/md_log.h>

modeldeploy::ImageData md_image_to_image_data(const MDImage* image) {
    cv::Mat cv_image = md_image_to_mat(image);
    return modeldeploy::ImageData::from_mat(&cv_image);
}

cv::Mat md_image_to_mat(const MDImage* image) {
    if (!image) {
        return {};
    }
    int cv_type = CV_8UC3;
    if (image->channels == 1) {
        cv_type = CV_8UC1;
    }
    else if (image->channels == 4) {
        cv_type = CV_8UC4;
    }
    auto mat = cv::Mat(image->height, image->width, cv_type, image->data);
    return mat;
}


MDImage* mat_to_md_image(const cv::Mat& mat) {
    const auto md_image = static_cast<MDImage*>(malloc(sizeof(MDImage)));
    // 设置宽度和高度
    md_image->width = mat.cols;
    md_image->height = mat.rows;
    md_image->channels = mat.channels();
    md_image->data = static_cast<unsigned char*>(malloc(mat.total() * mat.elemSize()));
    // 复制数据
    std::memcpy(md_image->data, mat.data, mat.total() * mat.elemSize());
    // 设置图像类型
    return md_image;
}

MDImage* image_data_to_md_image(const modeldeploy::ImageData& mat) {
    cv::Mat cv_image;
    mat.to_mat(&cv_image);
    return mat_to_md_image(cv_image);
}


void draw_rect_internal(cv::Mat& cv_image, const cv::Rect& rect, const cv::Scalar& cv_color, const double alpha) {
    cv::Mat overlay;
    cv_image.copyTo(overlay);
    cv::rectangle(overlay, {rect.x, rect.y, rect.width, rect.height}, cv_color, -1);
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    cv::rectangle(cv_image, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv_color, 1, cv::LINE_AA, 0);
}

void draw_polygon_internal(cv::Mat& cv_image, const std::vector<cv::Point>& points,
                           const cv::Scalar& color, const double alpha) {
    cv::Mat overlay;
    cv_image.copyTo(overlay);
    cv::fillPoly(overlay, points, color, cv::LINE_AA, 0);
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    cv::polylines(cv_image, points, true, color, 1, cv::LINE_AA, 0);
}

void draw_text_internal(cv::Mat& cv_image, const cv::Rect& rect, const std::string& text, const std::string& font_path,
                        const int font_size, const cv::Scalar& cv_color, const double alpha) {
    draw_rect_internal(cv_image, rect, cv_color, alpha);
    cv::FontFace font(font_path);
    const auto size = cv::getTextSize(cv::Size(0, 0), text, cv::Point(rect.x, rect.y), font, font_size);
    cv::rectangle(cv_image, size, cv_color, -1);
    cv::putText(cv_image, text, cv::Point(rect.x, rect.y - 3),
                cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
}

void draw_text_internal(cv::Mat& cv_image, const cv::Point& point, const std::string& text,
                        const std::string& font_path, const int font_size, const cv::Scalar& cv_color) {
    cv::FontFace font(font_path);
    cv::putText(cv_image, text, point, cv_color, font, font_size);
}

bool contains_substring(const std::string& str, const std::string& sub_str) {
    return str.find(sub_str) != std::string::npos;
}

std::string format_polygon(const MDPolygon polygon) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < polygon.size; i++) {
        os << "[" << polygon.data[i].x << "," << polygon.data[i].y << "]";
        if (i != polygon.size - 1) {
            os << ",";
        }
    }
    os << "]";
    return os.str();
}

std::string format_rect(const MDRect rect) {
    std::ostringstream os;
    os << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]";
    return os.str();
}

cv::Mat get_rotate_crop_image(const cv::Mat& src_image, const MDPolygon* polygon) {
    std::vector<cv::Point> points;
    points.reserve(polygon->size);
    for (int i = 0; i < polygon->size; i++) {
        points.emplace_back(polygon->data[i].x, polygon->data[i].y);
    }
    const cv::Rect bounding_box = cv::boundingRect(points);
    // 将多边形的点转换为相对于边界框的坐标
    std::vector<cv::Point2f> pointsf;
    pointsf.reserve(points.size());
    for (const auto& point : points) {
        pointsf.emplace_back(point.x - bounding_box.x, point.y - bounding_box.y);
    }

    // 计算目标图像的宽度和高度
    const int img_crop_width = bounding_box.width;
    const int img_crop_height = bounding_box.height;

    // 定义目标图像的四个角点
    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(static_cast<float>(img_crop_width), 0.);
    pts_std[2] = cv::Point2f(static_cast<float>(img_crop_width), static_cast<float>(img_crop_height));
    pts_std[3] = cv::Point2f(0.f, static_cast<float>(img_crop_height));

    // 计算透视变换矩阵
    const cv::Mat M = cv::getPerspectiveTransform(pointsf.data(), pts_std);
    // 裁剪并透视变换图像
    cv::Mat dst_img;
    cv::warpPerspective(src_image(bounding_box), dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);
    return dst_img;
}

// 注意开辟内存需要成对的销毁
void classification_result_2_c_results(
    const ClassifyResult& result,
    MDClassificationResults* c_results) {
    c_results->size = static_cast<int>(result.label_ids.size());
    c_results->data = new MDClassificationResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        c_results->data[i].label_id = result.label_ids[i];
        c_results->data[i].score = result.scores[i];
    }
}


void c_results_2_classification_result(
    const MDClassificationResults* c_results,
    ClassifyResult* result) {
    result->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        result->label_ids.emplace_back(c_results->data[i].label_id);
        result->scores.emplace_back(c_results->data[i].score);
    }
}


// 注意开辟内存需要成对的销毁
void detection_results_2_c_results(
    const std::vector<DetectionResult>& results,
    MDDetectionResults* c_results) {
    c_results->size = results.size();
    c_results->data = new MDDetectionResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        auto [x, y, width, height] = results[i].box;
        c_results->data[i].box = {
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(width),
            static_cast<int>(height)
        };
        c_results->data[i].score = results[i].score;
        c_results->data[i].label_id = results[i].label_id;
    }
}

void c_results_2_detection_results(
    const MDDetectionResults* c_results,
    std::vector<DetectionResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        Rect2f box = {
            static_cast<float>(c_results->data[i].box.x),
            static_cast<float>(c_results->data[i].box.y),
            static_cast<float>(c_results->data[i].box.width),
            static_cast<float>(c_results->data[i].box.height),
        };
        results->push_back({box, c_results->data[i].label_id, c_results->data[i].score});
    }
}

void obb_results_2_c_results(
    const std::vector<ObbResult>& results,
    MDObbResults* c_results) {
    c_results->size = results.size();
    c_results->data = new MDObbResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        auto rotated_box = results[i].rotated_box;
        c_results->data[i].rotated_box = {
            rotated_box.xc,
            rotated_box.yc,
            rotated_box.width,
            rotated_box.height,
            rotated_box.angle
        };
        c_results->data[i].score = results[i].score;
        c_results->data[i].label_id = results[i].label_id;
    }
}


void c_results_2_obb_results(
    const MDObbResults* c_results,
    std::vector<ObbResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        RotatedRect rotated_box = {
            c_results->data[i].rotated_box.xc,
            c_results->data[i].rotated_box.yc,
            c_results->data[i].rotated_box.width,
            c_results->data[i].rotated_box.height,
            c_results->data[i].rotated_box.angle,
        };
        results->push_back({rotated_box, c_results->data[i].label_id, c_results->data[i].score});
    }
}


void iseg_results_2_c_results(
    const std::vector<InstanceSegResult>& results,
    MDIsegResults* c_results) {
    c_results->size = results.size();
    c_results->data = new MDIsegResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        auto [x, y, width, height] = results[i].box;
        c_results->data[i].box = {
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(width),
            static_cast<int>(height)
        };
        c_results->data[i].score = results[i].score;
        c_results->data[i].label_id = results[i].label_id;

        if (results[i].mask.buffer.empty()) {
            c_results->data[i].mask.buffer = nullptr;
            c_results->data[i].mask.shape = nullptr;
            c_results->data[i].mask.num_dims = 0;
            c_results->data[i].mask.buffer_size = 0;
            return;
        }

        // 拷贝buffer
        const auto& mask_buffer = results[i].mask.buffer;
        c_results->data[i].mask.buffer = new char[mask_buffer.size()];
        std::copy(mask_buffer.begin(), mask_buffer.end(), c_results->data[i].mask.buffer);
        c_results->data[i].mask.buffer_size = static_cast<int>(mask_buffer.size());
        // 拷贝shape
        const auto& mask_shape = results[i].mask.shape;
        c_results->data[i].mask.shape = new int[mask_shape.size()];
        std::copy(mask_shape.begin(), mask_shape.end(), c_results->data[i].mask.shape);
        c_results->data[i].mask.num_dims = static_cast<int>(mask_shape.size());
    }
}


void c_results_2_iseg_results(
    const MDIsegResults* c_results,
    std::vector<InstanceSegResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        Rect2f box = {
            static_cast<float>(c_results->data[i].box.x),
            static_cast<float>(c_results->data[i].box.y),
            static_cast<float>(c_results->data[i].box.width),
            static_cast<float>(c_results->data[i].box.height),
        };
        Mask mask;
        if (c_results->data[i].mask.buffer != nullptr) {
            mask.buffer = std::vector<uint8_t>(
                c_results->data[i].mask.buffer,
                c_results->data[i].mask.buffer + c_results->data[i].mask.buffer_size);
            mask.shape = std::vector<int64_t>(
                c_results->data[i].mask.shape,
                c_results->data[i].mask.shape + c_results->data[i].mask.num_dims);
        }
        results->push_back({box, mask, c_results->data[i].label_id, c_results->data[i].score});
    }
}


void pose_results_2_c_results(
    const std::vector<PoseResult>& results,
    MDPoseResults* c_results) {
    c_results->size = results.size();
    c_results->data = new MDPoseResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        c_results->data[i].box = {
            static_cast<int>(results[i].box.x),
            static_cast<int>(results[i].box.y),
            static_cast<int>(results[i].box.width),
            static_cast<int>(results[i].box.height)
        };
        c_results->data[i].keypoints_size = static_cast<int>(results[i].keypoints.size());
        c_results->data[i].keypoints = new MDPoint3f[results[i].keypoints.size()];
        for (int j = 0; j < results[i].keypoints.size(); j++) {
            c_results->data[i].keypoints[j] = {
                results[i].keypoints[j].x,
                results[i].keypoints[j].y,
                results[i].keypoints[j].z
            };
        }
        c_results->data[i].label_id = results[i].label_id;
        c_results->data[i].score = results[i].score;
    }
}


void c_results_2_pose_results(
    const MDPoseResults* c_results,
    std::vector<PoseResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        auto box = Rect2f{
            static_cast<float>(c_results->data[i].box.x),
            static_cast<float>(c_results->data[i].box.y),
            static_cast<float>(c_results->data[i].box.width),
            static_cast<float>(c_results->data[i].box.height)
        };
        std::vector<Point3f> keypoints;
        keypoints.reserve(c_results->data[i].keypoints_size);
        for (int j = 0; j < c_results->data[i].keypoints_size; j++) {
            keypoints.emplace_back(c_results->data[i].keypoints[j].x,
                                   c_results->data[i].keypoints[j].y,
                                   c_results->data[i].keypoints[j].z);
        }
        results->push_back({box, keypoints, c_results->data[i].label_id, c_results->data[i].score});
    }
}


void ocr_result_2_c_results(
    const OCRResult& result,
    MDOCRResults* c_results) {
    c_results->size = static_cast<int>(result.boxes.size());
    c_results->data = new MDOCRResult[c_results->size];
    for (int i = 0; i < c_results->size; ++i) {
        auto text = result.text[i];
        // text
        c_results->data[i].text = strdup(text.c_str());
        //score
        c_results->data[i].score = result.rec_scores[i];
        // const 保证 data和size成员本身不被修改，但是不会限制data指向的内容不被修改
        //box
        MDPolygon polygon;
        polygon.size = 4;
        polygon.data = new MDPoint[polygon.size];
        polygon.data[0] = {result.boxes[i][0 * 2], result.boxes[i][0 * 2 + 1]};
        polygon.data[1] = {result.boxes[i][1 * 2], result.boxes[i][1 * 2 + 1]};
        polygon.data[2] = {result.boxes[i][2 * 2], result.boxes[i][2 * 2 + 1]};
        polygon.data[3] = {result.boxes[i][3 * 2], result.boxes[i][3 * 2 + 1]};
        c_results->data[i].box = polygon;
        c_results->data[i].table_boxes.size = 0;
        c_results->data[i].table_boxes.data = nullptr;
        c_results->data[i].table_structure = nullptr;

        // 注意table_boxes和box数量不一致，导致赋值会出问题，所以这里先不赋值table_boxes和table_structure
        if (!result.table_html.empty()) {
            // table_box
            // MDPolygon table_polygon;
            // table_polygon.size = 4;
            // table_polygon.data = new MDPoint[table_polygon.size];
            // table_polygon.data[0] = {result.table_boxes[i][0 * 2], result.table_boxes[i][0 * 2 + 1]};
            // table_polygon.data[1] = {result.table_boxes[i][1 * 2], result.table_boxes[i][1 * 2 + 1]};
            // table_polygon.data[2] = {result.table_boxes[i][2 * 2], result.table_boxes[i][2 * 2 + 1]};
            // table_polygon.data[3] = {result.table_boxes[i][3 * 2], result.table_boxes[i][3 * 2 + 1]};
            // c_results->data[i].table_boxes = table_polygon;
            // table_structure
            c_results->data[i].table_structure = strdup(result.table_structure[i].c_str());
        }
    }
    // table_html
    c_results->table_html = nullptr;
    if (!result.table_html.empty()) {
        c_results->table_html = strdup(result.table_html.c_str());
    }
}


void c_results_2_ocr_result(
    const MDOCRResults* c_results, OCRResult* result) {
    result->boxes.reserve(c_results->size);
    result->text.reserve(c_results->size);
    result->rec_scores.reserve(c_results->size);
    for (int i = 0; i < c_results->size; ++i) {
        result->boxes.emplace_back(
            std::array{
                c_results->data[i].box.data[0].x,
                c_results->data[i].box.data[0].y,
                c_results->data[i].box.data[1].x,
                c_results->data[i].box.data[1].y,
                c_results->data[i].box.data[2].x,
                c_results->data[i].box.data[2].y,
                c_results->data[i].box.data[3].x,
                c_results->data[i].box.data[3].y
            }
        );
        result->text.emplace_back(c_results->data[i].text);
        result->rec_scores.emplace_back(c_results->data[i].score);
        if (c_results->data[i].table_structure != nullptr) {
            // 注意table_boxes和box数量不一致，导致赋值会出问题，所以这里先不赋值table_boxes和table_structure
            // result->table_boxes.emplace_back(
            //     std::array{
            //         c_results->data[i].table_boxes.data[0].x,
            //         c_results->data[i].table_boxes.data[0].y,
            //         c_results->data[i].table_boxes.data[1].x,
            //         c_results->data[i].table_boxes.data[1].y,
            //         c_results->data[i].table_boxes.data[2].x,
            //         c_results->data[i].table_boxes.data[2].y,
            //         c_results->data[i].table_boxes.data[3].x,
            //         c_results->data[i].table_boxes.data[3].y
            //     }
            // );
            result->table_structure.emplace_back(c_results->data[i].table_structure);
        }
    }
    if (c_results->table_html != nullptr) {
        result->table_html = c_results->table_html;
    }
}


// 注意开辟内存需要成对的销毁
void detection_landmark_result_2_c_results(
    const std::vector<DetectionLandmarkResult>& results,
    MDDetectionLandmarkResults* c_results) {
    c_results->size = results.size();
    c_results->data = new MDDetectionLandmarkResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        c_results->data[i].box = {
            static_cast<int>(results[i].box.x),
            static_cast<int>(results[i].box.y),
            static_cast<int>(results[i].box.width),
            static_cast<int>(results[i].box.height)
        };
        c_results->data[i].landmarks_size = static_cast<int>(results[i].landmarks.size());
        c_results->data[i].landmarks = new MDPoint[results[i].landmarks.size()];
        for (int j = 0; j < results[i].landmarks.size(); j++) {
            c_results->data[i].landmarks[j] = {
                static_cast<int>(results[i].landmarks[j].x),
                static_cast<int>(results[i].landmarks[j].y),
            };
        }
        c_results->data[i].label_id = results[i].label_id;
        c_results->data[i].score = results[i].score;
    }
}

void c_results_2_detection_landmark_result(
    const MDDetectionLandmarkResults* c_results,
    std::vector<DetectionLandmarkResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        auto box = Rect2f{
            static_cast<float>(c_results->data[i].box.x),
            static_cast<float>(c_results->data[i].box.y),
            static_cast<float>(c_results->data[i].box.width),
            static_cast<float>(c_results->data[i].box.height)
        };
        std::vector<Point2f> landmarks;
        landmarks.reserve(c_results->data[i].landmarks_size);
        for (int j = 0; j < c_results->data[i].landmarks_size; j++) {
            landmarks.emplace_back(static_cast<float>(c_results->data[i].landmarks[j].x),
                                   static_cast<float>(c_results->data[i].landmarks[j].y));
        }
        results->push_back({box, landmarks, c_results->data[i].label_id, c_results->data[i].score});
    }
}

// 注意开辟内存需要成对的销毁
void lpr_results_2_c_results(
    const std::vector<LprResult>& results, MDLPRResults* c_results) {
    // 针对单纯的车牌识别模型
    if (results.empty()) {
        c_results->size = 1;
        c_results->data = new MDLPRResult[c_results->size];
        c_results->data[0].box = MDRect{0, 0, 0, 0};
        c_results->data[0].car_plate_color = strdup(results[0].car_plate_color.c_str());
        c_results->data[0].car_plate_str = strdup(results[0].car_plate_str.c_str());
        c_results->data[0].label_id = -1;
        c_results->data[0].score = 0.0f;
        c_results->data[0].landmarks_size = 0;
        c_results->data[0].landmarks = nullptr;
        return;
    }

    c_results->size = results.size();
    c_results->data = new MDLPRResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        c_results->data[i].box = {
            static_cast<int>(results[i].box.x),
            static_cast<int>(results[i].box.y),
            static_cast<int>(results[i].box.width),
            static_cast<int>(results[i].box.height)
        };
        c_results->data[i].landmarks_size = static_cast<int>(results[i].landmarks.size());
        c_results->data[i].landmarks = new MDPoint[results[i].landmarks.size()];
        for (int j = 0; j < results[i].landmarks.size(); j++) {
            c_results->data[i].landmarks[j] = {
                static_cast<int>(results[i].landmarks[j].x),
                static_cast<int>(results[i].landmarks[j].y),
            };
        }
        c_results->data[i].label_id = results[i].label_id;
        c_results->data[i].score = results[i].score;
        c_results->data[i].car_plate_str = strdup(results[i].car_plate_str.c_str());
        c_results->data[i].car_plate_color = strdup(results[i].car_plate_color.c_str());
    }
}

void c_results_2_lpr_results(
    const MDLPRResults* c_results, std::vector<LprResult>* results) {
    results->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        auto box = Rect2f{
            static_cast<float>(c_results->data[i].box.x),
            static_cast<float>(c_results->data[i].box.y),
            static_cast<float>(c_results->data[i].box.width),
            static_cast<float>(c_results->data[i].box.height)
        };
        std::vector<Point2f> landmarks;
        landmarks.reserve(c_results->data[i].landmarks_size);
        for (int j = 0; j < c_results->data[i].landmarks_size; j++) {
            landmarks.emplace_back(static_cast<float>(c_results->data[i].landmarks[j].x),
                                   static_cast<float>(c_results->data[i].landmarks[j].y));
        }
        results->push_back({
            box, landmarks, c_results->data[i].label_id, c_results->data[i].score,
            c_results->data[i].car_plate_str, c_results->data[i].car_plate_color
        });
    }
}


// 注意开辟内存需要成对的销毁
void face_recognizer_result_2_c_result(
    const FaceRecognitionResult& result, MDFaceRecognizerResult* c_result) {
    c_result->size = static_cast<int>(result.embedding.size());
    c_result->embedding = new float[c_result->size];
    std::copy(result.embedding.begin(), result.embedding.end(), c_result->embedding);
}


void c_result_2_face_recognizer_result(
    const MDFaceRecognizerResult* c_result,
    FaceRecognitionResult* result) {
    result->embedding.resize(c_result->size);
    result->embedding.assign(c_result->embedding, c_result->embedding + c_result->size);
}

// 注意开辟内存需要成对的销毁
void face_recognizer_results_2_c_results(
    const std::vector<FaceRecognitionResult>& results, MDFaceRecognizerResults* c_results) {
    c_results->size = static_cast<int>(results.size());
    c_results->data = new MDFaceRecognizerResult[results.size()];
    for (int i = 0; i < results.size(); i++) {
        c_results->data[i].embedding = new float[results[i].embedding.size()];
        std::copy(results[i].embedding.begin(), results[i].embedding.end(), c_results->data[i].embedding);
        c_results->data[i].size = static_cast<int>(results[i].embedding.size());
    }
}

void c_results_2_face_recognizer_results(
    const MDFaceRecognizerResults* c_results,
    std::vector<FaceRecognitionResult>* results) {
    results->resize(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        const size_t embedding_dim = c_results->data[i].size;
        results->at(i).embedding.resize(embedding_dim);
        results->at(i).embedding.assign(
            c_results->data[i].embedding,
            c_results->data[i].embedding + embedding_dim);
    }
}

void c_runtime_option_2_runtime_option(
    const MDRuntimeOption* c_option,
    modeldeploy::RuntimeOption* option) {
    option->set_cpu_thread_num(c_option->cpu_thread_num);
    option->ort_option.graph_optimization_level = c_option->graph_opt_level;
    option->ort_option.trt_engine_cache_path = c_option->trt_engine_cache_path;
    option->enable_fp16 = c_option->enable_fp16;
    option->enable_trt = c_option->enable_trt;
    option->device_id = c_option->device_id;
    option->device = static_cast<modeldeploy::Device>(c_option->device);
    option->backend = static_cast<modeldeploy::Backend>(c_option->backend);
    option->set_trt_min_shape(c_option->trt_min_shape);
    option->set_trt_opt_shape(c_option->trt_opt_shape);
    option->set_trt_max_shape(c_option->trt_max_shape);
    option->password = c_option->password;
}
