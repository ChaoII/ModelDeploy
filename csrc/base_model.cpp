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
        return infer(reused_input_tensors_, &reused_output_tensors_);
    }

    bool BaseModel::set_runtime(Runtime* clone_runtime) {
        runtime_ = std::unique_ptr<Runtime>(clone_runtime);
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
}
