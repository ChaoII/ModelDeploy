//
// Created by aichao on 2025/6/12.
//

#include "utils/utils.h"
#include "pybind/utils/utils.h"
#include "runtime/runtime.h"
#include "core/enum_variables.h"

namespace modeldeploy {
    std::vector<pybind11::array>
    infer_with_numpy_map(const Runtime& self, std::map<std::string, pybind11::array>& inputs) {
        // 原第一个 lambda 的实现
        std::vector<Tensor> _inputs;
        _inputs.reserve(inputs.size());
        for (auto& [name, array] : inputs) {
            // 获取 shape
            std::vector data_shape(array.shape(), array.shape() + array.ndim());
            // 获取 dtype 并分配 Tensor
            auto dtype = numpy_data_type_to_md_data_type(array.dtype());
            Tensor tensor;
            tensor.allocate(data_shape, dtype);
            tensor.set_name(name);
            tensor.from_external_memory(array.mutable_data(), tensor.shape(), tensor.dtype());
            // memcpy(tensor.inputs(), array.mutable_data(), array.nbytes());
            _inputs.push_back(std::move(tensor));
        }

        std::vector<Tensor> outputs(self.num_outputs());
        self.infer(_inputs, &outputs);
        std::vector<pybind11::array> results;
        results.reserve(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto numpy_dtype = md_data_type_to_numpy_data_type(outputs[i].dtype());
            results.emplace_back(numpy_dtype, outputs[i].shape());
            memcpy(results[i].mutable_data(), outputs[i].data(),
                   outputs[i].size() * Tensor::get_element_size(outputs[i].dtype()));
        }
        return results;
    }


    std::vector<Tensor> infer_with_tensor_map(const Runtime& self, const pybind11::dict& py_inputs) {
        // 原第二个 lambda 的实现
        std::vector<Tensor> _inputs;
        for (auto [tensor_name, tensor] : py_inputs) {
            auto name = pybind11::cast<std::string>(tensor_name);
            auto& t = pybind11::cast<Tensor&>(tensor); // 引用，不复制
            t.set_name(name);
            _inputs.push_back(std::move(t)); // 这里要看 Tensor 是否支持共享或 move
        }
        std::vector<Tensor> outputs;
        if (!self.infer(_inputs, &outputs)) {
            throw std::runtime_error("Failed to inference with Runtime.");
        }
        return outputs;
    }

    std::vector<Tensor> infer_with_tensor_vector(const Runtime& self, std::vector<Tensor>& inputs) {
        // 原第三个 lambda 的实现
        std::vector<Tensor> outputs;
        self.infer(inputs, &outputs);
        return outputs;
    }

    void bind_runtime(pybind11::module& m) {
        pybind11::enum_<Device>(m, "Device")
            .value("CPU", Device::CPU)
            .value("GPU", Device::GPU)
            .value("OPENCL", Device::OPENCL)
            .value("VULKAN", Device::VULKAN);

        pybind11::enum_<Backend>(m, "Backend")
            .value("NONE", NONE)
            .value("ORT", ORT)
            .value("TRT", ORT)
            .value("MNN", MNN);


        pybind11::class_<RuntimeOption>(m, "RuntimeOption")
            .def(pybind11::init())
            .def("set_model_path", &RuntimeOption::set_model_path, pybind11::arg("model_path"),
                 pybind11::arg("password") = "")
            .def("use_gpu", &RuntimeOption::use_gpu, pybind11::arg("device_id") = 0)
            .def("use_cpu", &RuntimeOption::use_cpu)
            .def("set_cpu_thread_num", &RuntimeOption::set_cpu_thread_num, pybind11::arg("thread_num") = -1)
            .def("use_ort_backend", &RuntimeOption::use_ort_backend)
            .def("use_mnn_backend", &RuntimeOption::use_mnn_backend)
            .def("use_trt_backend", &RuntimeOption::use_trt_backend)
            // 不暴露给python
            // .def_readwrite("ort_option", &RuntimeOption::ort_option)
            // .def("set_external_stream", &RuntimeOption::set_external_stream,
            //      pybind11::arg("stream"),
            //      pybind11::doc("A pointer to an external stream")
            // )
            // .def("set_external_raw_stream",
            //      [](RuntimeOption& self, size_t external_stream) {
            //          self.set_external_stream(reinterpret_cast<void*>(external_stream));
            //      })
            .def_readwrite("model_buffer", &RuntimeOption::model_buffer)
            .def_readwrite("model_file", &RuntimeOption::model_file)
            .def_readwrite("backend", &RuntimeOption::backend)
            .def_readwrite("cpu_thread_num", &RuntimeOption::cpu_thread_num)
            .def_readwrite("device_id", &RuntimeOption::device_id)
            .def_readwrite("device", &RuntimeOption::device)
            .def_readwrite("model_from_memory", &RuntimeOption::model_from_memory)
            .def_readwrite("enable_trt", &RuntimeOption::enable_trt)
            .def_readwrite("enable_fp16", &RuntimeOption::enable_fp16)
            .def_readwrite("password", &RuntimeOption::password);


        pybind11::class_<TensorInfo>(m, "TensorInfo")
            .def(pybind11::init())
            .def_readwrite("name", &TensorInfo::name)
            .def_readwrite("shape", &TensorInfo::shape)
            .def_readwrite("dtype", &TensorInfo::dtype)
            .def("__str__", &TensorInfo::str)
            .def("__repr__", &TensorInfo::str);

        pybind11::class_<Runtime>(m, "Runtime")
            .def(pybind11::init())
            .def("init", &Runtime::init)
            .def("infer", infer_with_numpy_map, pybind11::arg("inputs"))
            .def("infer", infer_with_tensor_map, pybind11::arg("inputs"))
            .def("infer", infer_with_tensor_vector, pybind11::arg("inputs"))
            .def("bind_input_tensor", &Runtime::bind_input_tensor, pybind11::arg("name"), pybind11::arg("input"))
            .def("bind_output_tensor", &Runtime::bind_output_tensor, pybind11::arg("name"), pybind11::arg("output"))
            .def("infer", [](Runtime& self) { self.infer(); })
            .def("get_output_tensor",
                 [](Runtime& self, const std::string& name) {
                     Tensor* output = self.get_output_tensor(name);
                     if (output == nullptr) {
                         return pybind11::cast(nullptr);
                     }
                     return pybind11::cast(*output);
                 }, pybind11::arg("name"))
            .def("num_inputs", &Runtime::num_inputs)
            .def("num_outputs", &Runtime::num_outputs)
            .def("get_input_info", &Runtime::get_input_info, pybind11::arg("index"))
            .def("get_output_info", &Runtime::get_output_info, pybind11::arg("index"))
            .def_readonly("option", &Runtime::option);
    }
}
