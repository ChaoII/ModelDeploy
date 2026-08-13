//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/sem/preprocessor.h"

#include <opencv2/opencv.hpp>

namespace modeldeploy::vision::detection {
    UltralyticsSemPreprocessor::UltralyticsSemPreprocessor() {
        size_ = {640, 640};
        padding_value_ = {114.0, 114.0, 114.0};
    }

    bool UltralyticsSemPreprocessor::preprocess(const ImageData& image, Tensor* output,
                                                LetterBoxRecord* letter_box_record) const {
        // 语义分割是像素级分类，必须用双线性插值（fused SIMD 最近邻会损失细节导致分类错误）
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        const int src_w = image.width();
        const int src_h = image.height();
        cv::Mat src_mat;
        image.to_mat(src_mat);  // BGR
        if (src_mat.empty() || src_w <= 0 || src_h <= 0) {
            return false;
        }
        const float scale = std::min(static_cast<float>(dst_h) / src_h,
                                     static_cast<float>(dst_w) / src_w);
        const int resize_w = static_cast<int>(std::round(src_w * scale));
        const int resize_h = static_cast<int>(std::round(src_h * scale));
        const int pad_w = (dst_w - resize_w) / 2;
        const int pad_h = (dst_h - resize_h) / 2;
        *letter_box_record = {static_cast<float>(src_w), static_cast<float>(src_h),
                              static_cast<float>(dst_w), static_cast<float>(dst_h),
                              static_cast<float>(pad_w), static_cast<float>(pad_h), scale};
        cv::Mat resized;
        cv::resize(src_mat, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat canvas(dst_h, dst_w, CV_8UC3, cv::Scalar(
            padding_value_[0], padding_value_[1], padding_value_[2]));
        resized.copyTo(canvas(cv::Rect(pad_w, pad_h, resize_w, resize_h)));
        // BGR -> RGB，/255，HWC -> CHW
        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
        output->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        float* dst = output->data_ptr<float>();
        const int plane = dst_h * dst_w;
        for (int y = 0; y < dst_h; ++y) {
            const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
            for (int x = 0; x < dst_w; ++x) {
                dst[0 * plane + y * dst_w + x] = row[x][0] / 255.0f;
                dst[1 * plane + y * dst_w + x] = row[x][1] / 255.0f;
                dst[2 * plane + y * dst_w + x] = row[x][2] / 255.0f;
            }
        }
        return true;
    }

    bool UltralyticsSemPreprocessor::run(
        const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
        std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images.size());
        outputs->resize(1);
        std::vector<Tensor> tensors(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            preprocess(images[i], &tensors[i], &(*letter_box_records)[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        } else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }
}
