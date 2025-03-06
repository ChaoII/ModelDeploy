//
// Created by aichao on 2025/2/8.
//
#include "capi/utils/internal/utils.h"
#include "capi/utils/md_draw_capi.h"

void md_draw_rect(MDImage* image, const MDRect* rect, const MDColor* color, const double alpha) {
    const auto cv_color = cv::Scalar(color->b, color->g, color->r);
    cv::Mat cv_image = md_image_to_mat(image);
    draw_rect_internal(cv_image, {rect->x, rect->y, rect->width, rect->height}, cv_color, alpha);
}

void md_draw_polygon(MDImage* image, const MDPolygon* polygon, const MDColor* color, const double alpha) {
    const auto cv_color = cv::Scalar(color->b, color->g, color->r);
    cv::Mat cv_image = md_image_to_mat(image);
    cv::Mat overlay;
    std::vector<cv::Point> points;
    points.reserve(polygon->size);
    for (int i = 0; i < polygon->size; ++i) {
        points.emplace_back(polygon->data[i].x, polygon->data[i].y);
    }
    draw_polygon_internal(cv_image, points, cv_color, alpha);
}


void md_draw_text(MDImage* image, const MDRect* rect, const char* text, const char* font_path, const int font_size,
                  const MDColor* color, const double alpha) {
    cv::Mat cv_image = md_image_to_mat(image);
    const auto cv_color = cv::Scalar(color->b, color->g, color->r);
    draw_text_internal(cv_image, {rect->x, rect->y, rect->width, rect->height}, text, font_path,
                       font_size, cv_color, alpha);
}
