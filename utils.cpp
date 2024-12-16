//
// Created by AC on 2024/12/16.
//
#include "utils.h"

cv::Mat wimage_to_mat(WImage *image) {
    WImageType type = image->type;
    int cv_type = CV_8UC3;
    switch (type) {
        case WImageType::WImageType_GRAY:
            cv_type = CV_8UC1;
            break;
        default:
            cv_type = CV_8UC3;
            break;
    }
    return {image->height, image->width, cv_type, image->data};
}


WImage *mat_to_wimage(const cv::Mat &mat) {
    auto wImage = (WImage *) malloc(sizeof(WImage));
    // 设置宽度和高度
    wImage->width = mat.cols;
    wImage->height = mat.rows;
    wImage->data = (unsigned char *) malloc(mat.total() * mat.elemSize());
    // 复制数据
    std::memcpy(wImage->data, mat.data, mat.total() * mat.elemSize());
    // 设置图像类型
    switch (mat.type()) {
        case CV_8UC1:
            wImage->type = WImageType_GRAY;
            break;
        case CV_8UC3:
            wImage->type = WImageType_BGR;
            break;
        default:
            return nullptr;
    }
    return wImage;
}


void print_rect(WRect rect) {
    std::cout << "rect: [" << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << "]" << std::endl;
}

void draw_rect(WImage *image, WRect rect, WColor color) {
    cv::Mat mat = wimage_to_mat(image);
    cv::rectangle(mat, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(color.b, color.g, color.r));
}

void draw_transparent_rectangle(cv::Mat &image, const std::vector<cv::Point> &points,
                                const cv::Scalar &color, double alpha) {
    cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    cv::Mat overlay;
    image.copyTo(overlay);
    cv::fillPoly(overlay, points, color, cv::LINE_AA, 0);
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
    cv::polylines(image, points, true, color, 1, cv::LINE_AA, 0);
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
}

void draw_text(WImage *image, WRect rect, const char *text, WColor color, double alpha) {
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    cv::Mat cv_image = wimage_to_mat(image);
    cv::Mat overlay;
    cv_image.copyTo(overlay);
    cv::rectangle(overlay, cv::Rect(rect.x, rect.y, rect.width, rect.height),
                  cv_color, -1);
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    cv::rectangle(cv_image, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv_color, 1, cv::LINE_AA, 0);
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGRA2BGR);
    cv::FontFace font("msyh.ttc");
    cv::putText(cv_image, text, cv::Point(rect.x, rect.y - 3),
                cv::Scalar(color.b, color.g, color.r), font, 20);
}

void show_image(WImage *image) {
    auto cv_image = wimage_to_mat(image);
    cv::imshow("image", cv_image);
    cv::waitKey(0);
}

bool contains_substring(const std::string &str, const std::string &sub_str) {
    return str.find(sub_str) != std::string::npos;
}

bool get_button_enable_status(WImage *image, int pix_threshold, double rate_threshold) {
    auto cv_image = wimage_to_mat(image);
    if (cv_image.channels() != 1) {
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2GRAY);
    }
    cv::Mat binaryImage;
    cv::threshold(cv_image, binaryImage, pix_threshold, 255, cv::THRESH_BINARY);
    int countAboveThreshold = static_cast<int>(cv_image.total()) - cv::countNonZero(binaryImage);
    // 计算像素所占百分比
    double percentage = (static_cast<double>(countAboveThreshold) / static_cast<int>(cv_image.total()));
    return percentage >= rate_threshold;
}

WImage *crop_image(WImage *image, WRect rect) {
    auto cv_image = wimage_to_mat(image);
    cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
    // 裁剪图像,并且让内存连续
    cv::Mat cropped_image = cv_image(roi).clone();
    return mat_to_wimage(cropped_image);
}

WImage *clone_image(WImage *image) {
    auto cv_image = wimage_to_mat(image).clone();
    return mat_to_wimage(cv_image);
}

WImage *from_compressed_bytes(const unsigned char *bytes, int size) {
    std::vector<unsigned char> buffer(bytes, bytes + size);
    cv::Mat img_decompressed = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return mat_to_wimage(img_decompressed);
}
