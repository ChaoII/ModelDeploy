//
// Created by AC on 2024/12/16.
//
#include "utils.h"

cv::Mat wimage_to_mat(WImage *image) {
    int cv_type = CV_8UC3;
    if (image->channels == 1) {
        cv_type = CV_8UC1;
    } else if (image->channels == 4) {
        cv_type = CV_8UC4;
    }
    return {image->height, image->width, cv_type, image->data};
}


WImage *mat_to_wimage(const cv::Mat &mat) {
    auto wImage = (WImage *) malloc(sizeof(WImage));
    // 设置宽度和高度
    wImage->width = mat.cols;
    wImage->height = mat.rows;
    wImage->channels = mat.channels();
    wImage->data = (unsigned char *) malloc(mat.total() * mat.elemSize());
    // 复制数据
    std::memcpy(wImage->data, mat.data, mat.total() * mat.elemSize());
    // 设置图像类型
    return wImage;
}


void print_rect(WRect rect) {
    std::cout << format_rect(rect) << std::endl;
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

std::string format_polygon(WPolygon polygon) {
    std::ostringstream os;
    os << "polygon: {";
    for (int i = 0; i < polygon.size; i++) {
        os << "[" << polygon.data[i].x << "," << polygon.data[i].y << "]";
        if (i != polygon.size - 1) {
            os << ",";
        }
    }
    os << "}";
    return os.str();
}

std::string format_rect(WRect rect) {
    std::ostringstream os;
    os << "WRect {" << "x: " << rect.x << ", " << "y: " << rect.y << ", "
       << "width: " << rect.width << ", " << "height: " << rect.height << "}";
    return os.str();
}

WModel *allocate_model() {
    return (WModel *) malloc(sizeof(WModel));
}

