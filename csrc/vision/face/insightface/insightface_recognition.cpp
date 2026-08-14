//
// insightface buffalo_l w600k_r50 ArcFace 识别实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/insightface_recognition.h"
#include "core/tensor.h"

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
        if (!embedding || kps.size() != 5) return false;
        cv::Mat src_mat;
        image.to_mat(src_mat);
        // norm_crop 到 112（arcface_dst 对齐）
        const cv::Mat warped = norm_crop(src_mat, kps, input_size_);
        // blobFromImages(1/127.5, (112,112), (127.5,127.5,127.5), swapRB=True)
        cv::Mat blob = make_blob_from_image(warped, 1.0f / 127.5f,
                                            cv::Scalar(127.5f, 127.5f, 127.5f),
                                            true /*swapRB*/);
        std::vector<Tensor> input_tensors(1);
        input_tensors[0].from_external_memory(blob.data, {1, 3, input_size_, input_size_},
                                              DataType::FP32, nullptr, Device::CPU,
                                              get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (timers) timers->infer_timer.start();
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();

        const float* out = static_cast<const float*>(output_tensors[0].data());
        const size_t dim = output_tensors[0].size();
        embedding->resize(dim);
        for (size_t i = 0; i < dim; ++i) (*embedding)[i] = out[i];
        return true;
    }

    std::unique_ptr<InsightFaceRecognition> InsightFaceRecognition::clone() const {
        auto clone_model = std::make_unique<InsightFaceRecognition>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

} // namespace modeldeploy::vision::face
