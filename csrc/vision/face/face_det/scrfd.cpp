//
// Created by aichao on 2025/2/20.
//


#include "core/md_log.h"
#include "vision/face/face_det/scrfd.h"


namespace modeldeploy::vision::face {
    Scrfd::Scrfd(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool Scrfd::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool Scrfd::predict(const ImageData& image, std::vector<KeyPointsResult>* result,
                        TimerArray* timers) {
        std::vector<std::vector<KeyPointsResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    std::unique_ptr<Scrfd> Scrfd::clone() const {
        auto clone_model = std::make_unique<Scrfd>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }
bool Scrfd::batch_predict(const std::vector<ImageData>& images,
                              std::vector<std::vector<KeyPointsResult>>* results,
                              TimerArray* timers) {
        std::vector<LetterBoxRecord> letter_box_records;
        // SCRFD 的 CPU 融合预处理把 plane(0) 当作 packed BGR 读取（NV12 只有 Y+UV 平面，
        // 按 BGR 读会越界段错误），故 NV12 一律先转 packed BGR；packed BGR 输入零拷贝直通。
        // 设备 NV12 先 toCpu() 回读到主机再转换，直接按设备内存解读会被误读。
        std::vector<ImageData> bgr_images;
        const std::vector<ImageData>* src_images = &images;
        bool need_convert = false;
        for (const auto& img : images) {
            if (img.format() == MdImageType::NV12) { need_convert = true; break; }
        }
        if (need_convert) {
            bgr_images.resize(images.size());
            for (size_t i = 0; i < images.size(); ++i) {
                if (images[i].format() != MdImageType::NV12) {
                    bgr_images[i] = images[i];
                    continue;
                }
                const ImageData* host = &images[i];
                ImageData host_owned;
                if (images[i].device() != Device::CPU) {
                    if (!images[i].toCpu(&host_owned)) {
                        MD_LOG_ERROR << "Scrfd: device NV12 toCpu failed." << std::endl;
                        return false;
                    }
                    host = &host_owned;
                }
                bgr_images[i] = ImageData::cvt_color(*host, ColorConvertType::CVT_NV122PKG_BGR);
                if (bgr_images[i].empty()) {
                    MD_LOG_ERROR << "Scrfd: NV12->BGR convert failed." << std::endl;
                    return false;
                }
            }
            src_images = &bgr_images;
        }
        std::vector<ImageData> _images = *src_images;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(&_images, &reused_input_tensors_, &letter_box_records)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, results, letter_box_records)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
}

