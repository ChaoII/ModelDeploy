//
// Created by AC on 2024/12/16.
//
#include "utils.h"
#include "utils_internal.h"

void md_print_rect(MDRect rect) {
    std::cout << format_rect(rect) << std::endl;
}

void md_draw_rect(MDImage *image, MDRect rect, MDColor color, double alpha) {
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    cv::Mat cv_image = md_image_to_mat(image);
    draw_rect_internal(cv_image, {rect.x, rect.y, rect.width, rect.height}, cv_color, alpha);
}

void md_draw_polygon(MDImage *image, MDPolygon *polygon, MDColor color, double alpha) {
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    cv::Mat cv_image = md_image_to_mat(image);
    cv::Mat overlay;
    std::vector<cv::Point> points(polygon->size);
    for (int i = 0; i < polygon->size; ++i) {
        points.emplace_back(polygon->data[i].x, polygon->data[i].y);
    }
    draw_polygon_internal(cv_image, points, cv_color, alpha);
}


void md_draw_text(MDImage *image, MDRect rect, const char *text, const char *font_path, int font_size, MDColor color,
               double alpha) {
    cv::Mat cv_image = md_image_to_mat(image);
    auto cv_color = cv::Scalar(color.b, color.g, color.r);
    draw_text_internal(cv_image, {rect.x, rect.y, rect.width, rect.height}, text, font_path,
                       font_size, cv_color, alpha);
}

void md_show_image(MDImage *image) {
    auto cv_image = md_image_to_mat(image);
    cv::imshow("image", cv_image);
    cv::waitKey(0);
}


bool md_get_button_enable_status(MDImage *image, int pix_threshold, double rate_threshold) {
    auto cv_image = md_image_to_mat(image);
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

MDImage *md_crop_image(MDImage *image, MDRect rect) {
    auto cv_image = md_image_to_mat(image);
    cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
    // 裁剪图像,并且让内存连续
    cv::Mat cropped_image = cv_image(roi).clone();
    return mat_to_md_image(cropped_image);
}

MDImage *md_clone_image(MDImage *image) {
    auto new_image = (MDImage *) malloc(sizeof(MDImage));
    new_image->width = image->width;
    new_image->height = image->height;
    new_image->channels = image->channels;
    auto total_bytes = image->width * image->height * image->channels * sizeof(unsigned char);
    new_image->data = (unsigned char *) malloc(total_bytes);
    std::memcpy(new_image->data, image->data, total_bytes);
    return new_image;
}

MDImage *md_from_compressed_bytes(const unsigned char *bytes, int size) {
    std::vector<unsigned char> buffer(bytes, bytes + size);
    cv::Mat img_decompressed = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return mat_to_md_image(img_decompressed);
}

MDImage *md_read_image(const char *path) {
    cv::Mat image = cv::imread(path);
    if (!image.empty()) {
        return mat_to_md_image(image);
    }
    return nullptr;
}

void md_free_image(MDImage *image) {
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

MDPoint md_get_center_point(MDRect rect) {
    return MDPoint{rect.x + rect.width / 2, rect.y + rect.height / 2};
}

MDModel *md_allocate_model() {
    return (MDModel *) malloc(sizeof(MDModel));
}

