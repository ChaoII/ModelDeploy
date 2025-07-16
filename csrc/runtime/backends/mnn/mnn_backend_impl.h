//
// Created by aichao on 2025/2/20.
//

#pragma once

#include <memory>
#include <vector>
#include <MNN/expr/Module.hpp>
#include "core/tensor.h"
#include "runtime/backends/backend.h"
#include "runtime/backends/mnn/option.h"

namespace modeldeploy {
    class MnnBackendImpl  {
    public:
        MnnBackendImpl() = default;
        ~MnnBackendImpl() = default;
        bool init(const RuntimeOption& runtime_option) ;

        [[nodiscard]] size_t num_inputs() const  {
            return inputs_desc_.size();
        }

        [[nodiscard]] size_t num_outputs() const  {
            return outputs_desc_.size();
        }

        TensorInfo get_input_info(int index) ;
        TensorInfo get_output_info(int index) ;
        std::vector<TensorInfo> get_input_infos() ;
        std::vector<TensorInfo> get_output_infos() ;
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) ;
        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const ;

    private:
        void build_option(const RuntimeOption& option);
        bool initialized_ = false;
        std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr_;
        MnnBackendOption option_;
        std::string model_buffer_;
        std::shared_ptr<MNN::Express::Module> net_;
        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
    };
} // namespace modeldeploy
