//
// Created by AC on 2024-12-24.
//

#include <fstream>
#include <random>
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"


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

#ifdef BUILD_FACE
SeetaImageData md_image_to_seeta_image(const MDImage* image) {
    const SeetaImageData seeta_image{
        image->width,
        image->height,
        image->channels,
        image->data
    };
    return seeta_image;
}
#endif

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
void detection_result_2_c_results(
    const DetectionResult& result,
    MDDetectionResults* c_results) {
    c_results->size = static_cast<int>(result.boxes.size());
    c_results->data = new MDDetectionResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        auto [xmin, ymin, xmax, ymax] = result.boxes[i];
        c_results->data[i].box = MDRect{
            static_cast<int>(xmin),
            static_cast<int>(ymin),
            static_cast<int>(xmax - xmin),
            static_cast<int>(ymax - ymin)
        };
        c_results->data[i].score = result.scores[i];
        c_results->data[i].label_id = result.label_ids[i];
    }
}


void c_results_2_detection_result(
    const MDDetectionResults* c_results,
    DetectionResult* result) {
    result->reserve(c_results->size);
    for (int i = 0; i < c_results->size; i++) {
        auto [x, y, width, height] = c_results->data[i].box;
        auto box = std::array<float, 4>{
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(x + width),
            static_cast<float>(y + height)
        };
        result->boxes.emplace_back(box);
        result->scores.emplace_back(c_results->data[i].score);
        result->label_ids.emplace_back(c_results->data[i].label_id);
    }
}

void ocr_result_2_c_results(
    const OCRResult& result,
    MDOCRResults* c_results) {
    c_results->size = static_cast<int>(result.boxes.size());
    c_results->data = new MDOCRResult[c_results->size];
    for (int i = 0; i < c_results->size; ++i) {
        auto text = result.text[i];
        c_results->data[i].text = strdup(text.c_str());
        c_results->data[i].score = result.rec_scores[i];
        // const 保证 data和size成员本身不被修改，但是不会限制data指向的内容不被修改
        MDPolygon polygon;
        polygon.size = 4;
        polygon.data = new MDPoint[polygon.size];
        polygon.data[0] = {result.boxes[i][0 * 2], result.boxes[i][0 * 2 + 1]};
        polygon.data[1] = {result.boxes[i][1 * 2], result.boxes[i][1 * 2 + 1]};
        polygon.data[2] = {result.boxes[i][2 * 2], result.boxes[i][2 * 2 + 1]};
        polygon.data[3] = {result.boxes[i][3 * 2], result.boxes[i][3 * 2 + 1]};
        c_results->data[i].box = polygon;
    }
}


void c_results_2_ocr_result(
    const MDOCRResults* c_results, OCRResult* result) {
    result->boxes.reserve(c_results->size);
    result->text.reserve(c_results->size);
    result->rec_scores.reserve(c_results->size);
    for (int i = 0; i < c_results->size; ++i) {
        result->boxes.emplace_back(
            std::array<int, 8>{
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
    }
}


// 注意开辟内存需要成对的销毁
void detection_landmark_result_2_c_results(
    const DetectionLandmarkResult& result,
    MDDetectionLandmarkResults* c_results) {
    c_results->size = static_cast<int>(result.boxes.size());
    c_results->data = new MDDetectionLandmarkResult[c_results->size];
    for (int i = 0; i < c_results->size; i++) {
        auto [xmin, ymin, xmax, ymax] = result.boxes[i];
        c_results->data[i].box = MDRect{
            static_cast<int>(xmin),
            static_cast<int>(ymin),
            static_cast<int>(xmax - xmin),
            static_cast<int>(ymax - ymin)
        };
        c_results->data[i].label_id = result.label_ids[i];
        c_results->data[i].score = result.scores[i];
        const int landmark_size = result.landmarks_per_instance;
        c_results->data[i].landmarks_size = landmark_size;
        c_results->data[i].landmarks = new MDPoint[landmark_size];
        for (int j = 0; j < landmark_size; j++) {
            c_results->data[i].landmarks[j] = MDPoint{
                static_cast<int>(result.landmarks[i * landmark_size + j][0]),
                static_cast<int>(result.landmarks[i * landmark_size + j][1])
            };
        }
    }
}

void c_results_2_detection_landmark_result(
    const MDDetectionLandmarkResults* c_results,
    DetectionLandmarkResult* result) {
    result->reserve(c_results->size);
    result->landmarks_per_instance = c_results->data[0].landmarks_size;
    for (int i = 0; i < c_results->size; i++) {
        result->boxes.emplace_back(std::array<float, 4>{
                static_cast<float>(c_results->data[i].box.x),
                static_cast<float>(c_results->data[i].box.y),
                static_cast<float>(c_results->data[i].box.x + c_results->data[i].box.width),
                static_cast<float>(c_results->data[i].box.y + c_results->data[i].box.height)
            }
        );
        result->label_ids.emplace_back(c_results->data[i].label_id);
        result->scores.emplace_back(c_results->data[i].score);
        for (int j = 0; j < c_results->data[i].landmarks_size; j++) {
            result->landmarks.emplace_back(
                std::array<float, 2>{
                    static_cast<float>(c_results->data[i].landmarks[j].x),
                    static_cast<float>(c_results->data[i].landmarks[j].y)
                }
            );
        }
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
    result->resize(c_result->size);
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
        results->at(i).resize(embedding_dim);
        results->at(i).embedding.assign(
            c_results->data[i].embedding,
            c_results->data[i].embedding + embedding_dim);
    }
}
