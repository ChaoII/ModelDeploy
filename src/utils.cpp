//
// Created by AC on 2024/12/16.
//
#include "utils.h"
#include "utils_internal.h"

void print_rect(WRect rect) {
    std::cout << format_rect(rect) << std::endl;
}

void draw_rect(WImage *image, WRect rect, WColor color, double alpha) {
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    cv::Mat cv_image = wimage_to_mat(image);
    draw_rect_internal(cv_image, {rect.x, rect.y, rect.width, rect.height}, cv_color, alpha);
}

void draw_polygon(WImage *image, WPolygon *polygon, WColor color, double alpha) {
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    cv::Mat cv_image = wimage_to_mat(image);
    cv::Mat overlay;
    std::vector<cv::Point> points(polygon->size);
    for (int i = 0; i < polygon->size; ++i) {
        points.emplace_back(polygon->data[i].x, polygon->data[i].y);
    }
    draw_polygon_internal(cv_image, points, cv_color, alpha);
}


void draw_text(WImage *image, WRect rect, const char *text, const char *font_path, int font_size, WColor color,
               double alpha) {
    cv::Mat cv_image = wimage_to_mat(image);
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    draw_text_internal(cv_image, {rect.x, rect.y, rect.width, rect.height}, text, font_path,
                       font_size, cv_color, alpha);
}

void show_image(WImage *image) {
    auto cv_image = wimage_to_mat(image);
    cv::imshow("image", cv_image);
    cv::waitKey(0);
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
    auto new_image = (WImage *) malloc(sizeof(WImage));
    new_image->width = image->width;
    new_image->height = image->height;
    new_image->channels = image->channels;
    auto total_bytes = image->width * image->height * image->channels * sizeof(unsigned char);
    new_image->data = (unsigned char *) malloc(total_bytes);
    std::memcpy(new_image->data, image->data, total_bytes);
    return new_image;
}

WImage *from_compressed_bytes(const unsigned char *bytes, int size) {
    std::vector<unsigned char> buffer(bytes, bytes + size);
    cv::Mat img_decompressed = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return mat_to_wimage(img_decompressed);
}

WImage *read_image(const char *path) {
    cv::Mat image = cv::imread(path);
    if (!image.empty()) {
        return mat_to_wimage(image);
    }
    return nullptr;
}

void free_wimage(WImage *image) {
    if (image != nullptr) {
        if (image->data != nullptr) {
            free(image->data);
            image->data = nullptr;  // 将指针置为 nullptr
            image->width = 0;
            image->height = 0;
            image->channels = 0;
        }
        free(image);
    }
}

WPoint get_center_point(WRect rect) {
    return WPoint{rect.x + rect.width / 2, rect.y + rect.height / 2};
}

WModel *allocate_model() {
    return (WModel *) malloc(sizeof(WModel));
}

