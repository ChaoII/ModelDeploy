#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "csrc/core/tensor.h"
#include "csrc/core/md_log.h"
#include "csrc/runtime/runtime_option.h"

namespace modeldeploy {
    struct TensorInfo {
        std::string name;
        std::vector<int> shape;
        DataType dtype;
    };

    class BaseBackend {
    public:
        bool initialized_ = false;

        BaseBackend() = default;

        virtual ~BaseBackend() = default;

        [[nodiscard]] virtual bool is_initialized() const { return initialized_; }

        virtual bool init(const RuntimeOption& option) {
            MD_LOG_ERROR << "Not Implement for "
                << option.backend << " in "
                << option.device << "."
                << std::endl;
            return false;
        }

        [[nodiscard]] virtual size_t num_inputs() const = 0;

        [[nodiscard]] virtual size_t num_outputs() const = 0;

        virtual TensorInfo get_input_info(int index) = 0;

        virtual TensorInfo get_output_info(int index) = 0;

        virtual std::vector<TensorInfo> get_input_infos() = 0;

        virtual std::vector<TensorInfo> get_output_infos() = 0;

        [[nodiscard]] virtual std::map<std::string, std::string> get_custom_meta_data() const = 0;

        virtual bool infer(std::vector<Tensor>& inputs,
                           std::vector<Tensor>* outputs) = 0;
    };
}
