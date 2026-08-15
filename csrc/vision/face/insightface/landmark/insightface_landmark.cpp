//
// insightface buffalo_l landmark 模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/landmark/insightface_landmark.h"
#include "vision/face/insightface/face_align_utils.h"
#include "vision/processors/processor_factory.h"
#include <cstring>

namespace modeldeploy::vision::face {

    InsightFaceLandmark::InsightFaceLandmark(const std::string& model_file,
                                             const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceLandmark::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceLandmark::predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                                            std::vector<std::array<float, 2>>* landmarks,
                                            TimerArray* timers) {
        if (!landmarks) return false;
        cv::Mat M;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, bbox, &M, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_2d(reused_output_tensors_, inv_M, input_size_[0], landmarks)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    bool InsightFaceLandmark::predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                                           std::vector<std::array<float, 3>>* landmarks,
                                           std::array<float, 3>* pose,
                                           TimerArray* timers) {
        if (!landmarks) return false;
        cv::Mat M;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, bbox, &M, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_3d(reused_output_tensors_, inv_M, input_size_[0], landmarks, pose)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    bool InsightFaceLandmark::batch_predict_2d106(
        const ImageData& image, const std::vector<std::array<float, 4>>& bboxes,
        std::vector<std::vector<std::array<float, 2>>>* landmarks_list,
        TimerArray* timers) {
        if (!landmarks_list || bboxes.empty()) return false;
        const size_t n = bboxes.size();
        landmarks_list->resize(n);
        if (n == 1) {
            return predict_2d106(image, bboxes[0], &(*landmarks_list)[0], timers);
        }
        const int H = input_size_[0];
        const int W = input_size_[1];
        const size_t plane = static_cast<size_t>(3) * H * W;
        // 每张脸单图预处理到临时 tensor + 保存 M，memcpy 合并到 batch
        std::vector<Tensor> singles(n);
        std::vector<cv::Mat> Ms(n);
        Tensor batch_tensor({static_cast<int64_t>(n), 3, H, W}, DataType::FP32, Device::CPU);
        float* batch_data = batch_tensor.data_ptr<float>();
        if (timers) timers->pre_timer.start();
        for (size_t i = 0; i < n; ++i) {
            if (!preprocessor_.run(image, bboxes[i], &Ms[i], &singles[i])) return false;
            std::memcpy(batch_data + i * plane, singles[i].data(), plane * sizeof(float));
        }
        if (timers) timers->pre_timer.stop();
        batch_tensor.set_name(get_input_info(0).name);
        std::vector<Tensor> input_tensors{batch_tensor};
        std::vector<Tensor> output_tensors;
        if (timers) timers->infer_timer.start();
        if (!infer(input_tensors, &output_tensors)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        // 输出按 batch 行切分（batch 维度），逐脸逆仿射
        const size_t total = output_tensors[0].size();
        const size_t per = total / n; // 每脸输出元素数（单脸时 = total）
        for (size_t i = 0; i < n; ++i) {
            // 切分第 i 张脸的输出：从 data 偏移 i*per
            Tensor face_out(static_cast<char*>(output_tensors[0].data()) + i * per * sizeof(float),
                            {static_cast<int64_t>(per)}, output_tensors[0].dtype(), Device::CPU);
            const cv::Mat inv_M = invert_affine_transform(Ms[i]);
            std::vector<Tensor> one{face_out};
            if (!postprocessor_.run_2d(one, inv_M, input_size_[0], &(*landmarks_list)[i])) return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }

    bool InsightFaceLandmark::batch_predict_3d68(
        const ImageData& image, const std::vector<std::array<float, 4>>& bboxes,
        std::vector<std::vector<std::array<float, 3>>>* landmarks_list,
        std::vector<std::array<float, 3>>* poses,
        TimerArray* timers) {
        if (!landmarks_list || bboxes.empty()) return false;
        const size_t n = bboxes.size();
        landmarks_list->resize(n);
        if (poses) poses->resize(n);
        if (n == 1) {
            std::array<float, 3> pose{0, 0, 0};
            bool ok = predict_3d68(image, bboxes[0], &(*landmarks_list)[0],
                                   poses ? &pose : nullptr, timers);
            if (ok && poses) (*poses)[0] = pose;
            return ok;
        }
        const int H = input_size_[0];
        const int W = input_size_[1];
        const size_t plane = static_cast<size_t>(3) * H * W;
        std::vector<Tensor> singles(n);
        std::vector<cv::Mat> Ms(n);
        Tensor batch_tensor({static_cast<int64_t>(n), 3, H, W}, DataType::FP32, Device::CPU);
        float* batch_data = batch_tensor.data_ptr<float>();
        if (timers) timers->pre_timer.start();
        for (size_t i = 0; i < n; ++i) {
            if (!preprocessor_.run(image, bboxes[i], &Ms[i], &singles[i])) return false;
            std::memcpy(batch_data + i * plane, singles[i].data(), plane * sizeof(float));
        }
        if (timers) timers->pre_timer.stop();
        batch_tensor.set_name(get_input_info(0).name);
        std::vector<Tensor> input_tensors{batch_tensor};
        std::vector<Tensor> output_tensors;
        if (timers) timers->infer_timer.start();
        if (!infer(input_tensors, &output_tensors)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        const size_t total = output_tensors[0].size();
        const size_t per = total / n;
        for (size_t i = 0; i < n; ++i) {
            Tensor face_out(static_cast<char*>(output_tensors[0].data()) + i * per * sizeof(float),
                            {static_cast<int64_t>(per)}, output_tensors[0].dtype(), Device::CPU);
            const cv::Mat inv_M = invert_affine_transform(Ms[i]);
            std::array<float, 3> pose{0, 0, 0};
            std::vector<Tensor> one{face_out};
            if (!postprocessor_.run_3d(one, inv_M, input_size_[0], &(*landmarks_list)[i],
                                       poses ? &pose : nullptr)) return false;
            if (poses) (*poses)[i] = pose;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceLandmark> InsightFaceLandmark::clone() const {
        auto clone_model = std::make_unique<InsightFaceLandmark>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
