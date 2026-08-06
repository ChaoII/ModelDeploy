//
// Created by aichao on 2025/2/20.
//


#include "core/md_log.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/utils.h"

#ifdef ENABLE_SOPHGO
#include "runtime/backends/sophgo/sophgo_backend.h"
#include "vision/processors/sophgo/sophgo_processor_backend.h"
#include "vision/processors/sophgo/bmcv_bridge.h"
#endif

namespace modeldeploy::vision::detection {
    UltralyticsDet::UltralyticsDet(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsDet::initialize() {
        if (!init_runtime()) {
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
#ifdef ENABLE_SOPHGO
        // Sophgo：把 backend 的 bm_handle 注入 processor，保证 BMCV 预处理与 bmrt 推理同一设备上下文
        if (runtime_option.device == Device::TPU && runtime_option.backend == Backend::SOPHGO) {
            auto* sop_back = dynamic_cast<SophgoBackend*>(get_backend());
            auto* sop_pre = dynamic_cast<vision::SophgoProcessorBackend*>(
                preprocessor_.get_processor_backend().get());
            if (sop_back && sop_pre) {
                void* h = sop_back->get_bm_handle();
                if (h) sop_pre->use_external_handle(h);
            }
        }
#endif
        return true;
    }

    bool UltralyticsDet::predict(const ImageData& image, std::vector<DetectionResult>* result,
                                 TimerArray* timers) {
        std::vector<std::vector<DetectionResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }


    bool UltralyticsDet::batch_predict(const std::vector<ImageData>& images,
                                       std::vector<std::vector<DetectionResult>>* results,
                                       TimerArray* timers) {
#ifdef ENABLE_SOPHGO
        // 设备内存零拷贝推理：BMCV 预处理直接写入 bmrt_tensor 分配的输入设备内存（attach），
        // 再 bmrt_launch_tensor 推理（参考官方 SOPHON-DEMO YOLOv8_plus_det 实现）。
        // 关键：必须先把 out_img attach 到 input 设备内存再 convert（否则 launch 读到空内存挂 TPU）。
        if (runtime_option.backend == Backend::SOPHGO && images.size() == 1) {
            auto* sop_pre = dynamic_cast<vision::SophgoProcessorBackend*>(
                preprocessor_.get_processor_backend().get());
            auto* sop_back = dynamic_cast<SophgoBackend*>(get_backend());
            const auto& sz = preprocessor_.get_size();
            if (sop_pre && sop_back && sz.size() == 2 && !images[0].empty()) {
                const LetterBoxRecord lb = utils::cal_letter_box_param(
                    {images[0].width(), images[0].height()}, sz);
                float ox, oy, sx, sy;
                utils::letter_box_to_fused_params(lb, &ox, &oy, &sx, &sy);
                const std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
                const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
                const auto& pv = preprocessor_.get_padding_value();
                const float pad = pv.empty() ? 114.0f : pv[0];
                // 获取缓存的输入设备内存（bmrt_tensor 分配，bm_device_mem_t*）
                void* input_mem = sop_back->get_input_device_mem();
                if (!input_mem) {
                    MD_LOG_WARN << "Sophgo zero-copy input mem unavailable, fallback to normal path." << std::endl;
                    goto fallback_normal;
                }
                int dw = sz[0], dh = sz[1];
                bool ok = false;
                if (timers) timers->pre_timer.start();
                try {
                    ok = sop_pre->fused_preprocess_device(
                        images[0], &out_img_, input_mem, &dw, &dh,
                        ox, oy, sx, sy, alpha, beta,
                        /*swap_rb*/true, pad / 255.0f);
                } catch (const std::exception& e) {
                    MD_LOG_ERROR << "Sophgo zero-copy preprocess threw: " << e.what() << std::endl;
                    ok = false;
                }
                if (timers) timers->pre_timer.stop();
                if (ok) {
                    const std::vector<int64_t> shape = {1, 3, dh, dw};
                    if (timers) timers->infer_timer.start();
                    try {
                        // BMCV 结果已写入 input_mem（bm_device_mem_t*，bmrt_tensor 分配）。
                        // 用 from_external_memory 包装为 Device::TPU Tensor（不拷贝、不拥有），
                        // 统一走 infer()，其识别 TPU 输入跳过 s2d 直接 launch（零拷贝）。
                        reused_input_tensors_.clear();
                        reused_input_tensors_.resize(1);
                        reused_input_tensors_[0].from_external_memory(
                            input_mem, shape, DataType::FP32,
                            [](void*) {} /*deleter: 设备内存由 backend 管理*/, Device::TPU,
                            get_input_info(0).name);
                        ok = infer(reused_input_tensors_, &reused_output_tensors_);
                    } catch (const std::exception& e) {
                        MD_LOG_ERROR << "Sophgo zero-copy infer threw: " << e.what() << std::endl;
                        ok = false;
                    }
                    if (timers) timers->infer_timer.stop();
                }
                vision::md_bmcv_image_destroy(out_img_);
                out_img_ = nullptr;
                if (ok) {
                    if (timers) timers->post_timer.start();
                    std::vector<std::vector<DetectionResult>> tmp;
                    bool pos = false;
                    try {
                        pos = postprocessor_.run(reused_output_tensors_, &tmp, {lb});
                    } catch (const std::exception& e) {
                        MD_LOG_ERROR << "Sophgo zero-copy postprocess threw: " << e.what() << std::endl;
                        pos = false;
                    }
                    if (timers) timers->post_timer.stop();
                    results->resize(1);
                    if (pos && !tmp.empty()) (*results)[0] = std::move(tmp[0]);
                    return pos;
                }
                MD_LOG_WARN << "Sophgo zero-copy path failed, fallback to normal path." << std::endl;
            }
        }
    fallback_normal:
#endif
        std::vector<LetterBoxRecord> letter_box_records;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, &letter_box_records)) {
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

    bool UltralyticsDet::predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                      int width, int height, int step_y, int step_uv,
                                      std::vector<DetectionResult>* result,
                                      LetterBoxRecord* letter_box_record,
                                      TimerArray* timers) {
        if (!src_y || !src_uv || !result) return false;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(src_y, src_uv, {width, height}, step_y, step_uv,
                               &reused_input_tensors_[0], letter_box_record)) {
            MD_LOG_ERROR << "Failed to preprocess the NV12 input." << std::endl;
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
        std::vector<std::vector<DetectionResult>> results;
        if (!postprocessor_.run(reused_output_tensors_, &results, {*letter_box_record})) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        if (!results.empty()) *result = std::move(results[0]);
        return true;
    }

    std::unique_ptr<UltralyticsDet> UltralyticsDet::clone() const {
        auto clone_model = std::make_unique<UltralyticsDet>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
}

