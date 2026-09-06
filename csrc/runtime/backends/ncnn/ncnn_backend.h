#pragma once

#include <memory>
#include <vector>
#include <ncnn/net.h>
#include "core/tensor.h"
#include "runtime/backends/backend.h"
#include "runtime/backends/ncnn/option.h"

namespace modeldeploy {
    class NcnnBackend : public BaseBackend {
    public:
        NcnnBackend() = default;
        ~NcnnBackend() override;
        bool init(const RuntimeOption& runtime_option) override;
        [[nodiscard]] size_t num_inputs() const override { return input_names_.size(); }
        [[nodiscard]] size_t num_outputs() const override { return output_names_.size(); }
        TensorInfo get_input_info(int index) override;
        TensorInfo get_output_info(int index) override;
        std::vector<TensorInfo> get_input_infos() override;
        std::vector<TensorInfo> get_output_infos() override;
        bool infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) override;
        std::unique_ptr<BaseBackend> clone(const RuntimeOption& runtime_option,
                                           void* stream = nullptr,
                                           int device_id = -1) override;

    private:
        // 与存活中的其它 Vulkan NcnnBackend 共享的 ncnn GPU 会话所有权。
        // 以 shared_ptr 管理，最后一个持有者释放时自动拆除 GPU 实例（见 ncnn_backend.cpp）。
        std::shared_ptr<void> vk_session_;
        std::unique_ptr<ncnn::Net> net_;
        NcnnBackendOption option_;
        RuntimeOption saved_option_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
        std::vector<TensorInfo> input_info_;
        std::vector<TensorInfo> output_info_;
        // init 时若所有输入形状已由 param 声明（正维），用零数据跑一次 dummy forward 补全输出形状。
        void infer_output_info();
    };
} // namespace modeldeploy
