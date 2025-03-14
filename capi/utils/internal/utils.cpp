//
// Created by AC on 2024-12-24.
//

#include <fstream>
#include <random>
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

cv::Scalar get_random_color() {
    std::random_device rd; // 获取随机数种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister算法生成随机数
    std::uniform_int_distribution dis(0, 255); // 定义随机数范围为1到255
    return {
        static_cast<double>(dis(gen)),
        static_cast<double>(dis(gen)),
        static_cast<double>(dis(gen))
    };
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


// using os_string = std::filesystem::path::string_type;
#ifdef _WIN32
#include <Windows.h>
using os_string = std::wstring;
#else
using os_string = std::string;
#endif
os_string to_osstring(std::string_view utf8_str) {
#ifdef _WIN32
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), nullptr, 0);
    os_string result(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.data(), static_cast<int>(utf8_str.size()), result.data(), len);
    return result;
#else
    return std::string(utf8_str);
#endif
}

bool read_binary_from_file(const std::string& path, std::string* contents) {
    if (!contents) {
        return false;
    }
    auto& result = *contents;
    result.clear();

    std::ifstream file(to_osstring(path), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    if (const auto file_size = file.tellg(); file_size != -1) {
        result.resize(file_size);
        file.seekg(0, std::ios::beg);
        file.read(result.data(), file_size);
    }
    else {
        // no size available, read to EOF
        constexpr auto chunk_size = 4096;
        std::string chunk(chunk_size, 0);
        while (!file.fail()) {
            file.read(chunk.data(), chunk_size);
            result.insert(result.end(), chunk.data(), chunk.data() + file.gcount());
        }
    }
    return true;
}
