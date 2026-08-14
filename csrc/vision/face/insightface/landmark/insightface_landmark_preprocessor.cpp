//
// insightface buffalo_l landmark 前处理实现。
// 与 python face_align.transform 逐值对齐：warpAffine（INTER_LINEAR + BORDER_CONSTANT 0）。
// warpAffine 的连续坐标 + 显式 border 语义无法用 fused 的坐标越界 pad 精确表达，故用 OpenCV。
//
#include "vision/face/insightface/landmark/insightface_landmark_preprocessor.h"
#include <algorithm>
#include <cstring>

namespace modeldeploy::vision::face {

    InsightFaceLandmarkPreprocessor::InsightFaceLandmarkPreprocessor() {
        size_ = {192, 192};
    }

    bool InsightFaceLandmarkPreprocessor::run(const ImageData& image, const std::array<float, 4>& bbox,
                                              cv::Mat* M, Tensor* output) const {
        const int dst = size_[0];
        const float w = bbox[2] - bbox[0];
        const float h = bbox[3] - bbox[1];
        const float cx = (bbox[2] + bbox[0]) / 2.0f;
        const float cy = (bbox[3] + bbox[1]) / 2.0f;
        const float scale = static_cast<float>(dst) / (std::max(w, h) * 1.5f);
        // python transform（rotation=0）: dst = (src - center)*scale + dst/2
        cv::Mat M2x3 = cv::Mat::zeros(2, 3, CV_64F);
        M2x3.at<double>(0, 0) = scale;
        M2x3.at<double>(1, 1) = scale;
        M2x3.at<double>(0, 2) = (dst / 2.0) - cx * scale;
        M2x3.at<double>(1, 2) = (dst / 2.0) - cy * scale;
        // 双线性 warpAffine（对齐 python cv2.warpAffine INTER_LINEAR + BORDER_CONSTANT 0）
        cv::Mat src_mat;
        image.to_mat(src_mat);
        cv::Mat warped;
        cv::warpAffine(src_mat, warped, M2x3, cv::Size(dst, dst),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        // 模型输入 0-255 RGB（内部归一化），仅 swapRB
        std::vector<float> blob(static_cast<size_t>(3) * dst * dst);
        const uint8_t* src = warped.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB
            float* plane = blob.data() + static_cast<size_t>(c) * dst * dst;
            for (int i = 0; i < dst * dst; ++i) plane[i] = static_cast<float>(src[i * 3 + src_c]);
        }
        output->allocate({1, 3, dst, dst}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        if (M) *M = M2x3;
        return true;
    }

} // namespace modeldeploy::vision::face
