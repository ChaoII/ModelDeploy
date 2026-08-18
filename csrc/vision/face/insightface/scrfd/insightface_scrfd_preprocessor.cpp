//
// insightface buffalo_l det_10g 前处理实现。
// 与 python SCRFD._detect_candidates 逐值对齐：等比例双线性 resize + 左上放置 + pad 0。
// 注：det 的"resize 到 new_w/new_h + 显式 pad"语义无法用 fused 的坐标越界 pad 表达，
// 且需对齐 python cv2.resize INTER_LINEAR（half-pixel），故 preprocessor 内用 OpenCV。
//
#include "core/md_log.h"
#include "vision/face/insightface/scrfd/insightface_scrfd_preprocessor.h"
#include <opencv2/opencv.hpp>
#include <cstring>

namespace modeldeploy::vision::face {

    InsightFaceDetPreprocessor::InsightFaceDetPreprocessor() {
        size_ = {640, 640};
    }

    bool InsightFaceDetPreprocessor::run(const ImageData& image, Tensor* output,
                                         LetterBoxRecord* letter_box_record) const {
        const int src_w = image.width();
        const int src_h = image.height();
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        const float im_ratio = static_cast<float>(src_h) / src_w;
        const float model_ratio = static_cast<float>(dst_h) / dst_w;
        int new_w, new_h;
        if (im_ratio > model_ratio) {
            new_h = dst_h;
            new_w = static_cast<int>(std::round(new_h / im_ratio));
        } else {
            new_w = dst_w;
            new_h = static_cast<int>(std::round(new_w * im_ratio));
        }
        // 双线性 resize + 左上放置 + pad 0（python cv2.resize INTER_LINEAR + det_img 左上）
        cv::Mat src_mat;
        image.asMat(&src_mat);
        cv::Mat resized;
        cv::resize(src_mat, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat det_img(dst_h, dst_w, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));
        // 记录 det_scale（后处理缩放回原图）= new_h / src_h
        letter_box_record->ipt_w = static_cast<float>(src_w);
        letter_box_record->ipt_h = static_cast<float>(src_h);
        letter_box_record->scale = static_cast<float>(new_h) / src_h;
        letter_box_record->pad_w = 0.0f;
        letter_box_record->pad_h = 0.0f;
        letter_box_record->out_w = static_cast<float>(dst_w);
        letter_box_record->out_h = static_cast<float>(dst_h);
        // blob：(x-127.5)/128 + BGR2RGB（对齐 cv2.dnn.blobFromImage）
        std::vector<float> blob(static_cast<size_t>(3) * dst_h * dst_w);
        const uint8_t* src = det_img.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB
            float* plane = blob.data() + static_cast<size_t>(c) * dst_h * dst_w;
            for (int i = 0; i < dst_h * dst_w; ++i) {
                plane[i] = (static_cast<float>(src[i * 3 + src_c]) - 127.5f) * (1.0f / 128.0f);
            }
        }
        output->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        return true;
    }

    bool InsightFaceDetPreprocessor::run(const std::vector<ImageData>& images, Tensor* output,
                                         std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) return false;
        letter_box_records->resize(images.size());
        if (images.size() == 1) {
            return run(images[0], output, &(*letter_box_records)[0]);
        }
        const int n = static_cast<int>(images.size());
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        std::vector<Tensor> singles(n);
        for (int i = 0; i < n; ++i) {
            if (!run(images[i], &singles[i], &(*letter_box_records)[i])) return false;
        }
        output->allocate({n, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        for (int i = 0; i < n; ++i) {
            std::memcpy(static_cast<char*>(output->data()) + static_cast<size_t>(i) * 3 * dst_h * dst_w * sizeof(float),
                        singles[i].data(), static_cast<size_t>(3) * dst_h * dst_w * sizeof(float));
        }
        return true;
    }

} // namespace modeldeploy::vision::face
