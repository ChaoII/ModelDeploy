//
// insightface buffalo_l w600k_r50 ArcFace 识别模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/recognition/insightface_recognition.h"
#include <cstring>

namespace modeldeploy::vision::face {

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

    bool InsightFaceRecognition::batch_predict(
        const ImageData& image,
        const std::vector<std::vector<std::array<float, 2>>>& kps_list,
        std::vector<std::vector<float>>* embeddings,
        TimerArray* timers) {
        if (!embeddings || kps_list.empty()) return false;
        const size_t n = kps_list.size();
        embeddings->resize(n);
        // 单脸：走单图路径（等价）
        if (n == 1) {
            return predict(image, kps_list[0], &(*embeddings)[0], timers);
        }
        const int H = preprocessor_.input_size_;
        const int W = preprocessor_.input_size_;
        const size_t plane = static_cast<size_t>(3) * H * W;
        Tensor batch_tensor({static_cast<int64_t>(n), 3, H, W}, DataType::FP32, Device::CPU);
        float* batch_data = batch_tensor.data_ptr<float>();
        // 每张脸单图预处理到临时 tensor，memcpy 合并到 batch
        std::vector<Tensor> singles(n);
        for (size_t i = 0; i < n; ++i) {
            if (!preprocessor_.run(image, kps_list[i], &singles[i])) return false;
            std::memcpy(batch_data + i * plane, singles[i].data(), plane * sizeof(float));
        }
        batch_tensor.set_name(get_input_info(0).name);
        std::vector<Tensor> input_tensors{batch_tensor};
        std::vector<Tensor> output_tensors;
        if (timers) timers->infer_timer.start();
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to batch inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        // 输出 [N,512]，按行切分
        const size_t dim = output_tensors[0].size() / n;
        const float* out = static_cast<const float*>(output_tensors[0].data());
        for (size_t i = 0; i < n; ++i) {
            (*embeddings)[i].resize(dim);
            std::memcpy((*embeddings)[i].data(), out + i * dim, dim * sizeof(float));
        }
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
