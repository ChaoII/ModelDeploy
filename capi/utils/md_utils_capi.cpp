//
// Created by AC on 2024/12/16.
//
#include <filesystem>
#include "capi/common/md_micro.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/utils/internal/utils.h"

namespace fs = std::filesystem;


MDKeyValuePair md_create_key_value_pair(const int key, const char* value) {
    MDKeyValuePair pair;
    pair.key = key;
    pair.value = strdup(value);
    return pair;
}

void md_free_md_map(MDMapData* c_map) {
    for (int i = 0; i < c_map->size; i++) {
        free(c_map->data[i].value);
    }
    free(c_map->data);
    c_map->size = 0;
}

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

MDRect md_create_rect_from_polygon(const MDPolygon* polygon) {
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (int i = 0; i < polygon->size; i++) {
        if (polygon->data[i].x < min_x) min_x = polygon->data[i].x;
        if (polygon->data[i].x > max_x) max_x = polygon->data[i].x;
        if (polygon->data[i].y < min_y) min_y = polygon->data[i].y;
        if (polygon->data[i].y > max_y) max_y = polygon->data[i].y;
    }
    return MDRect{
        static_cast<int>(min_x),
        static_cast<int>(min_y),
        static_cast<int>(max_x - min_x),
        static_cast<int>(max_y - min_y)
    };
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

MDRuntimeOption md_create_default_runtime_option() {
    // 这里是允许返回栈上构造的结构体
    return MDRuntimeOption{
        "",
        "",
        "",
        "./trt_engine",
        0,
        -1,
        0,
        0,
        MD_DEVICE_CPU,
        MD_BACKEND_ORT,
        -1,
        ""
    };
}
