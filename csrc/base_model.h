//
// Created by aichao on 2025/2/20.
//

#pragma once
#include "csrc/runtime/ort.h"

namespace modeldeploy {
    class BaseModel {
    public:
        virtual ~BaseModel() = default;

        [[nodiscard]] virtual std::string name() const { return "NameUndefined"; }

        virtual bool infer(std::vector<MDTensor>& input_tensors,
                           std::vector<MDTensor>* output_tensors);

        virtual bool infer();

        virtual int num_inputs() { return runtime_->num_inputs(); }

        virtual int num_outputs() { return runtime_->num_outputs(); }

        virtual TensorInfo get_input_info(int index) {
            return runtime_->get_input_info(index);
        }

        virtual TensorInfo get_output_info(int index) {
            return runtime_->get_output_info(index);
        }

        [[nodiscard]] virtual bool initialized() const {
            return runtime_initialized_ && initialized_;
        }

        virtual void release_reused_buffer() {
            reused_input_tensors_.shrink_to_fit();
            reused_output_tensors_.shrink_to_fit();
        }

        virtual bool set_runtime(OrtBackend* clone_runtime) {
            runtime_ = std::unique_ptr<OrtBackend>(clone_runtime);
            return true;
        }

    protected:
        virtual bool init_runtime();
        bool initialized_ = false;
        RuntimeOption runtime_option_;
        std::vector<MDTensor> reused_input_tensors_;
        std::vector<MDTensor> reused_output_tensors_;

    private:
        std::shared_ptr<OrtBackend> runtime_;
        bool runtime_initialized_ = false;
    };
}
