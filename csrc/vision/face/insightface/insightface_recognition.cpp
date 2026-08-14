//
// insightface buffalo_l w600k_r50 ArcFace 识别实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/insightface_recognition.h"
#include "core/tensor.h"
#include <cstring>

namespace modeldeploy::vision::face {

    // ==================== Preprocessor ====================

    InsightFaceRecPreprocessor::InsightFaceRecPreprocessor() = default;

    void InsightFaceRecPreprocessor::make_blob(const cv::Mat& warped, float* dst) const {
        // 与 cv2.dnn.blobFromImages(1/127.5, (112,112), (127.5,), swapRB=True) 一致
        const int h = warped.rows, w = warped.cols;
        const uint8_t* src = warped.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB: 输出通道0=R(原BGR[2])
            const float scale = 1.0f / 127.5f;
            const float mean = 127.5f;
            float* plane = dst + static_cast<size_t>(c) * h * w;
            for (int i = 0; i < h * w; ++i) {
                plane[i] = (static_cast<float>(src[i * 3 + src_c]) - mean) * scale;
            }
        }
    }

    bool InsightFaceRecPreprocessor::run(const ImageData& image,
                                         const std::vector<std::array<float, 2>>& kps,
                                         Tensor* output) const {
        if (kps.size() != 5) return false;
        cv::Mat src_mat;
        image.to_mat(src_mat);
        // norm_crop：Umeyama 相似变换 + warpAffine（含旋转，架构无此 fused 算子）
        const cv::Mat warped = norm_crop(src_mat, kps, input_size_);
        // blob (x-127.5)/127.5 + swapRB -> [1,3,112,112] FP32
        std::vector<float> blob(static_cast<size_t>(3) * input_size_ * input_size_);
        make_blob(warped, blob.data());
        output->allocate({1, 3, input_size_, input_size_}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        return true;
    }

    // ==================== Postprocessor ====================

    bool InsightFaceRecPostprocessor::run(const std::vector<Tensor>& infer_results,
                                          std::vector<float>* embedding) {
        if (infer_results.empty()) return false;
        const float* out = static_cast<const float*>(infer_results[0].data());
        const size_t dim = infer_results[0].size();
        embedding->resize(dim);
        for (size_t i = 0; i < dim; ++i) (*embedding)[i] = out[i];
        return true;
    }

    // ==================== Model ====================

    InsightFaceRecognition::InsightFaceRecognition(const std::string& model_file,
                                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceRecognition::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    bool InsightFaceRecognition::predict(const ImageData& image,
                                         const std::vector<std::array<float, 2>>& kps,
                                         std::vector<float>* embedding,
                                         TimerArray* timers) {
        if (!embedding) return false;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, kps, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, embedding)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceRecognition> InsightFaceRecognition::clone() const {
        auto clone_model = std::make_unique<InsightFaceRecognition>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
