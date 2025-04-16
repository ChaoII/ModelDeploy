//
// Created by aichao on 2025/2/20.
//

#include "base_model.h"
#include "core/md_log.h"


namespace modeldeploy {
    bool BaseModel::init_runtime() {
        if (runtime_initialized_) {
            std::cerr << "The model is already initialized, cannot be initialized again." << std::endl;
            return false;
        }
        runtime_ = std::make_shared<OrtBackend>();
        if (!runtime_->init(runtime_option_)) {
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
}
