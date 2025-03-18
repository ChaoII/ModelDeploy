//
// Created by AC on 2024/12/16.
//
#include <filesystem>
#include "capi/utils/md_utils_capi.h"
#include "capi/utils/internal/utils.h"

namespace fs = std::filesystem;

void md_print_rect(const MDRect* rect) {
    std::cout << format_rect(*rect) << std::endl;
}


bool md_get_button_enable_status(const MDImage* image, const int pix_threshold, const double rate_threshold) {
    auto cv_image = md_image_to_mat(image);
    if (cv_image.channels() != 1) {
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2GRAY);
    }
    cv::Mat binaryImage;
    cv::threshold(cv_image, binaryImage, pix_threshold, 255, cv::THRESH_BINARY);
    const int countAboveThreshold = static_cast<int>(cv_image.total()) - cv::countNonZero(binaryImage);
    // 计算像素所占百分比
    const double percentage = static_cast<double>(countAboveThreshold) / static_cast<double>(cv_image.total());
    return percentage >= rate_threshold;
}

MDPolygon* md_create_polygon_from_rect(const MDRect* rect) {
    auto* polygon = static_cast<MDPolygon*>(malloc(sizeof(MDPolygon)));
    polygon->data = static_cast<MDPoint*>(malloc(sizeof(MDPoint) * 4));
    polygon->data[0] = MDPoint{rect->x, rect->y};
    polygon->data[1] = MDPoint{rect->x + rect->width, rect->y};
    polygon->data[2] = MDPoint{rect->x + rect->width, rect->y + rect->height};
    polygon->data[3] = MDPoint{rect->x, rect->y + rect->height};
    polygon->size = 4;
    return polygon;
}


MDPoint md_get_center_point(const MDRect* rect) {
    return MDPoint{rect->x + rect->width / 2, rect->y + rect->height / 2};
}


void print_polygon(const MDPolygon* polygon) {
    std::string res = "{";
    for (int i = 0; i < polygon->size; i++) {
        res += "{" + std::to_string(polygon->data[i].x) + "," + std::to_string(polygon->data[i].y) + "}";
        if (i != polygon->size - 1) {
            res += ",";
        }
    }
    res += "}";
    std::cout << res << std::endl;
}

void md_free_polygon(MDPolygon* polygon) {
    polygon->size = 0;
    free(polygon->data);
}
