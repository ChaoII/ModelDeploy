//
// Created by aichao on 2025/2/8.
//

#include "csrc/core/md_log.h"
#include "capi/utils/md_image_capi.h"
#include "csrc/utils/utils.h"
#include "capi/utils/internal/utils.h"


void md_show_image(MDImage* image) {
    const auto cv_image = md_image_to_mat(image);
    cv::imshow("image", cv_image);
    cv::waitKey(0);
}

MDImage md_crop_image(MDImage* image, const MDRect* rect) {
    const auto cv_image = md_image_to_mat(image);
    const cv::Rect roi(rect->x, rect->y, rect->width, rect->height);
    // 裁剪图像,并且让内存连续
    const cv::Mat cropped_image = cv_image(roi).clone();
    return *mat_to_md_image(cropped_image);
}

MDImage md_clone_image(const MDImage* image) {
    MDImage new_image;
    new_image.width = image->width;
    new_image.height = image->height;
    new_image.channels = image->channels;
    const auto total_bytes = image->width * image->height * image->channels * sizeof(unsigned char);
    new_image.data = static_cast<unsigned char*>(malloc(total_bytes));
    std::memcpy(new_image.data, image->data, total_bytes);
    return new_image;
}

MDImage md_from_compressed_bytes(const unsigned char* bytes, const int size) {
    const std::vector buffer(bytes, bytes + size);
    const cv::Mat img_decompressed = cv::imdecode(buffer, cv::IMREAD_COLOR);
    return *mat_to_md_image(img_decompressed);
}

MDImage md_from_base64_str(const char* base64_str) {
    const std::vector buffer = modeldeploy::base64_decode(base64_str);
    const cv::Mat img_decode = cv::imdecode(buffer, cv::IMREAD_COLOR);
    // 检查图像是否成功解码
    if (img_decode.empty()) {
        MD_LOG_ERROR << "Failed to decode image from base64 string" << std::endl;
        return {};
    }
    // 将 cv::Mat 转换为 MDImage
    return *mat_to_md_image(img_decode);
}


MDImage md_read_image(const char* path) {
    const cv::Mat image = cv::imread(path);
    return *mat_to_md_image(image);
}

MDImage md_read_image_from_device(int device_id, int frame_width, int frame_height, bool is_save_file) {
#ifdef _WIN32
    cv::VideoCapture cap(device_id, cv::CAP_DSHOW);
#else
    cv::VideoCapture cap(device_id);
#endif
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
    if (!cap.isOpened()) {
        MD_LOG_WARN << "Open video stream or file failed" << std::endl;
        return {};
    }
    cv::Mat image;
    int black = 0;
    while (true) {
        if (const bool ret = cap.read(image); !ret) {
            MD_LOG_ERROR << "No frame" << std::endl;
            continue;
        }
        if (image.empty()) {
            MD_LOG_ERROR << "Empty frame" << std::endl;
            continue;
        }
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        if (cv::countNonZero(channels[0]) == 0) {
            MD_LOG_WARN << "All black frame: " << ++black;
            continue;
        }
        break;
    }
    cap.release();
    if (is_save_file) {
        cv::imwrite("capture.jpg", image);
        MD_LOG_INFO << "Save file to [capture.jpg]" << std::endl;
    }
    return *mat_to_md_image(image);
}

void md_save_image(MDImage* image, const char* path) {
    cv::imwrite(path, md_image_to_mat(image));
}


void md_free_image(MDImage* image) {
    if (image == nullptr) return;
    if (image->data != nullptr) {
        free(image->data);
        image->data = nullptr; // 将指针置为 nullptr
        image->width = 0;
        image->height = 0;
        image->channels = 0;
    }
}
