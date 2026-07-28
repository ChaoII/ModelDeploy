//
// Created by aichao on 2025/2/20.
//

#include "base_model.h"
#include "core/md_log.h"

namespace modeldeploy {
    bool BaseModel::init_runtime() {
        if (runtime_initialized_) {
            MD_LOG_ERROR << "The model is already initialized, cannot be initialized again." << std::endl;
            return false;
        }
        runtime_ = std::make_shared<Runtime>();
        if (!runtime_->init(runtime_option)) {
            return false;
        }
        runtime_initialized_ = true;
        return true;
    }

    bool BaseModel::infer(std::vector<Tensor>& input_tensors,
                          std::vector<Tensor>* output_tensors) {
        const auto ret = runtime_->infer(input_tensors, output_tensors);
        return ret;
    }

    bool BaseModel::infer() {
        // 同一实例并发推理检测 + 警告（无参 infer 使用 reused_tensors_，非线程安全）
        if (infer_busy_.test_and_set()) {
            static std::once_flag flag;
            std::call_once(flag, []() {
                MD_LOG_WARN << "检测到同一 BaseModel 实例并发 infer()！"
                            << "这是假并发，不会提升性能。"
                            << "请使用 clone() 为每个线程创建独立实例。"
                            << std::endl;
            });
        }
        std::lock_guard<std::mutex> lock(infer_mtx_);
        struct BusyGuard { std::atomic_flag& f; ~BusyGuard() { f.clear(); } } bg{infer_busy_};
        return infer(reused_input_tensors_, &reused_output_tensors_);
    }

    bool BaseModel::set_runtime(std::unique_ptr<Runtime> clone_runtime) {
        runtime_ = std::move(clone_runtime);
        return true;
    }

    size_t BaseModel::num_inputs() { return runtime_->num_inputs(); }

    size_t BaseModel::num_outputs() { return runtime_->num_outputs(); }

    TensorInfo BaseModel::get_input_info(const int index) const {
        return runtime_->get_input_info(index);
    }

    TensorInfo BaseModel::get_output_info(const int index) const {
        return runtime_->get_output_info(index);
    }

    bool BaseModel::is_initialized() const {
        return runtime_initialized_ && initialized_;
    }

    std::map<std::string, std::string> BaseModel::get_custom_meta_data() {
        return runtime_->get_custom_meta_data();
    }


    std::unordered_map<int, std::string> BaseModel::get_label_map(const std::string& label_map_key) {
        const auto meta_data = get_custom_meta_data();
        if (meta_data.find(label_map_key) == meta_data.end()) {
            return {};
        }
        const auto& label_string = meta_data.at(label_map_key);
        return parse_label_map(label_string);
    }
}
