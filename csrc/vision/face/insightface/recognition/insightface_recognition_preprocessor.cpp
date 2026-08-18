//
// insightface buffalo_l w600k_r50 前处理实现。
//
#include "vision/face/insightface/recognition/insightface_recognition_preprocessor.h"
#include "vision/face/insightface/face_align_utils.h"
#include "core/tensor.h"
#include <cstring>

namespace modeldeploy::vision::face {

    InsightFaceRecPreprocessor::InsightFaceRecPreprocessor() = default;

    void InsightFaceRecPreprocessor::make_blob(const cv::Mat& warped, float* dst) const {
        // 与 cv2.dnn.blobFromImages(1/127.5, (112,112), (127.5,), swapRB=True) 一致
        const int h = warped.rows, w = warped.cols;
        const uint8_t* src = warped.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB: 输出通道0=R(原BGR[2])
            float* plane = dst + static_cast<size_t>(c) * h * w;
            for (int i = 0; i < h * w; ++i) {
                plane[i] = (static_cast<float>(src[i * 3 + src_c]) - 127.5f) * (1.0f / 127.5f);
            }
        }
    }

    bool InsightFaceRecPreprocessor::run(const ImageData& image,
                                         const std::vector<std::array<float, 2>>& kps,
                                         Tensor* output) const {
        if (kps.size() != 5) return false;
        cv::Mat src_mat;
        image.asMat(&src_mat);
        // norm_crop：Umeyama 相似变换 + warpAffine（含旋转）
        const cv::Mat warped = norm_crop(src_mat, kps, input_size_);
        std::vector<float> blob(static_cast<size_t>(3) * input_size_ * input_size_);
        make_blob(warped, blob.data());
        output->allocate({1, 3, input_size_, input_size_}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        return true;
    }

} // namespace modeldeploy::vision::face
