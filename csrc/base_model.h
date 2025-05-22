//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <string>
#include <map>
#include "csrc/core/tensor.h"
#include "csrc/core/md_decl.h"
#include "csrc/runtime/runtime.h"


namespace modeldeploy {
    class MODELDEPLOY_CXX_EXPORT BaseModel {
    public:
        virtual ~BaseModel() = default;

        [[nodiscard]] virtual std::string name() const { return "NameUndefined"; }

        virtual bool infer(std::vector<Tensor>& input_tensors,
                           std::vector<Tensor>* output_tensors);

        virtual bool infer();

        virtual size_t num_inputs();

        virtual size_t num_outputs();

        [[nodiscard]] TensorInfo get_input_info(int index) const;

        [[nodiscard]] TensorInfo get_output_info(int index) const;

        [[nodiscard]] virtual bool is_initialized() const;

        virtual void release_reused_buffer() {
            reused_input_tensors_.shrink_to_fit();
            reused_output_tensors_.shrink_to_fit();
        }

        virtual bool set_runtime(Runtime* clone_runtime) {
            runtime_ = std::unique_ptr<Runtime>(clone_runtime);
            return true;
        }

        virtual std::map<std::string, std::string> get_custom_meta_data();

    protected:
        virtual bool init_runtime();

        bool initialized_ = false;
        RuntimeOption runtime_option_{};
        std::vector<Tensor> reused_input_tensors_;
        std::vector<Tensor> reused_output_tensors_;

    private:
        std::shared_ptr<Runtime> runtime_;
        bool runtime_initialized_ = false;
    };
}
