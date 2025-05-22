//
// Created by aichao on 2025/5/22.
//
#pragma once
#include "csrc/runtime/backends/backend.h"
#include "csrc/core/tensor.h"
#include "csrc/runtime/runtime_option.h"


/** \brief All C++ FastDeploy APIs are defined inside this namespace
*
*/
namespace modeldeploy {
    /*! @brief Runtime object used to inference the loaded model on different devices
     */
    struct MODELDEPLOY_CXX_EXPORT Runtime {
    public:
        /// Intialize a Runtime object with RuntimeOption
        bool init(const RuntimeOption& _option);

        /** \brief Inference the model by the input data, and write to the output
         *
         * \param[in] input_tensors Notice the FDTensor::name should keep same with the model's input
         * \param[in] output_tensors Inference results
         * \return true if the inference successed, otherwise false
         */
        bool infer(std::vector<Tensor>& input_tensors,
                   std::vector<Tensor>* output_tensors);

        /** \brief No params inference the model.
         *
         *  the input and output data need to pass through the BindInputTensor and GetOutputTensor interfaces.
         */
        bool infer();

        /** \brief Get number of inputs
         */
        [[nodiscard]] size_t num_inputs() const { return backend_->num_inputs(); }
        /** \brief Get number of outputs
         */
        [[nodiscard]] size_t num_outputs() const { return backend_->num_outputs(); }
        /** \brief Get input information by index
         */
        TensorInfo get_input_info(int index);
        /** \brief Get output information by index
         */
        TensorInfo get_output_info(int index);
        /** \brief Get all the input information
         */
        std::vector<TensorInfo> get_input_infos();
        /** \brief Get all the output information
         */
        std::vector<TensorInfo> get_output_infos();
        /** \brief Bind FDTensor by name, no copy and share input memory
         */
        void bind_input_tensor(const std::string& name, Tensor& input);

        /** \brief Bind FDTensor by name, no copy and share output memory.
         *  Please make share the correctness of tensor shape of output.
         */
        void bind_output_tensor(const std::string& name, Tensor& output);

        /** \brief Get output FDTensor by name, no copy and share backend output memory
         */
        Tensor* get_output_tensor(const std::string& name);

        [[nodiscard]] std::map<std::string, std::string> get_custom_meta_data() const {
            return backend_->get_custom_meta_data();
        }

        RuntimeOption option;


        [[nodiscard]] bool is_initialized() const { return backend_->is_initialized(); }

    private:
        void CreateOrtBackend();
        std::unique_ptr<BaseBackend> backend_;
        std::vector<Tensor> input_tensors_;
        std::vector<Tensor> output_tensors_;
    };
} // namespace fastdeploy
