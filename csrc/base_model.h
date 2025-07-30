//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <string>
#include <map>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "runtime/runtime.h"
#include "utils/benchmark.h"


namespace modeldeploy {
    class MODELDEPLOY_CXX_EXPORT BaseModel {
    public:
        virtual ~BaseModel() = default;

        [[nodiscard]] virtual std::string name() const { return "NameUndefined"; }

        virtual bool infer(std::vector<Tensor>& input_tensors,
                           std::vector<Tensor>* output_tensors);

        virtual bool infer();

        virtual Runtime* clone_runtime() { return runtime_->clone(); }

        virtual bool set_runtime(Runtime* clone_runtime);

        virtual size_t num_inputs();

        virtual size_t num_outputs();

        [[nodiscard]] TensorInfo get_input_info(int index) const;

        [[nodiscard]] TensorInfo get_output_info(int index) const;

        [[nodiscard]] virtual bool is_initialized() const;

        virtual void release_reused_buffer() {
            reused_input_tensors_.shrink_to_fit();
            reused_output_tensors_.shrink_to_fit();
        }

        virtual std::map<std::string, std::string> get_custom_meta_data();

        virtual std::unordered_map<int, std::string> get_label_map(const std::string& label_map_key);

        RuntimeOption runtime_option{};

    protected:
        virtual bool init_runtime();
        bool initialized_ = false;
        std::vector<Tensor> reused_input_tensors_;
        std::vector<Tensor> reused_output_tensors_;

    private:
        std::shared_ptr<Runtime> runtime_;
        bool runtime_initialized_ = false;
    };
}
