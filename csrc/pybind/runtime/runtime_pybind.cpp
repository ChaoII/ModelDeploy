//
// Created by aichao on 2025/6/12.
//

#include "csrc/utils/utils.h"
#include "csrc/pybind/utils/utils.h"
#include "csrc/runtime/runtime.h"

namespace modeldeploy {
    std::vector<pybind11::array> infer_with_numpy_map(Runtime& self, std::map<std::string, pybind11::array>& data) {
        // 原第一个 lambda 的实现
        std::vector<Tensor> inputs;
        inputs.reserve(data.size());
        for (auto& [name, array] : data) {
            // 获取 shape
            std::vector<int64_t> data_shape(array.shape(), array.shape() + array.ndim());
            // 获取 dtype 并分配 Tensor
            auto dtype = numpy_data_type_to_md_data_type(array.dtype());
            Tensor tensor;
            tensor.allocate(data_shape, dtype);
            tensor.set_name(name);
            // 拷贝数据（TODO: 后续考虑使用 SetExternalData 避免拷贝）
            tensor.from_external_memory(array.mutable_data(), tensor.shape(), tensor.dtype());
            // memcpy(tensor.data(), array.mutable_data(), array.nbytes());
            inputs.push_back(std::move(tensor));
        }

        std::vector<Tensor> outputs(self.num_outputs());
        self.infer(inputs, &outputs);
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


    std::vector<Tensor> infer_with_tensor_map(Runtime& self, std::map<std::string, Tensor>& data) {
        // 原第二个 lambda 的实现
        std::vector<Tensor> inputs;
        inputs.reserve(data.size());
        for (auto iter = data.begin(); iter != data.end(); ++iter) {
            Tensor tensor;
            tensor.from_external_memory(iter->second.data(), iter->second.shape(),
                                        iter->second.dtype());
            tensor.set_name(iter->first);
            inputs.push_back(tensor);
        }
        std::vector<Tensor> outputs;
        if (!self.infer(inputs, &outputs)) {
            throw std::runtime_error("Failed to inference with Runtime.");
        }
        return outputs;
    }

    std::vector<Tensor> infer_with_tensor_vector(Runtime& self, std::vector<Tensor>& inputs) {
        // 原第三个 lambda 的实现
        std::vector<Tensor> outputs;
        self.infer(inputs, &outputs);
        return outputs;
    }

    void bind_runtime(pybind11::module& m) {
        pybind11::enum_<Device>(m, "Device")
            .value("CPU", CPU)
            .value("GPU", GPU);

        pybind11::enum_<Backend>(m, "Backend")
            .value("NONE", NONE)
            .value("ORT", ORT);


        pybind11::class_<RuntimeOption>(m, "RuntimeOption")
            .def(pybind11::init())
            .def("set_model_path", &RuntimeOption::set_model_path)
            .def("use_gpu", &RuntimeOption::use_gpu)
            .def("use_cpu", &RuntimeOption::use_cpu)
            .def("set_ort_graph_opt_level", &RuntimeOption::set_ort_graph_opt_level)
            .def_readwrite("model_buffer", &RuntimeOption::model_buffer)
            // .def_readwrite("ort_option", &RuntimeOption::ort_option)
            .def("set_external_stream", &RuntimeOption::set_external_stream,
                 pybind11::arg("stream"),
                 pybind11::doc("A pointer to an external stream")
            )
            .def("set_external_raw_stream",
                 [](RuntimeOption& self, size_t external_stream) {
                     self.set_external_stream(reinterpret_cast<void*>(external_stream));
                 })
            .def("set_cpu_thread_num", &RuntimeOption::set_cpu_thread_num)
            .def("use_ort_backend", &RuntimeOption::use_ort_backend)
            .def_readwrite("model_file", &RuntimeOption::model_file)
            .def_readwrite("backend", &RuntimeOption::backend)
            .def_readwrite("cpu_thread_num", &RuntimeOption::cpu_thread_num)
            .def_readwrite("device_id", &RuntimeOption::device_id)
            .def_readwrite("device", &RuntimeOption::device)
            .def_readwrite("model_from_memory", &RuntimeOption::model_from_memory)
            .def_readwrite("enable_trt", &RuntimeOption::enable_trt)
            .def_readwrite("enable_fp16", &RuntimeOption::enable_fp16);

        pybind11::enum_<DataType>(m, "DataType")
            .value("FP32", DataType::FP32)
            .value("FP64", DataType::FP64)
            .value("INT32", DataType::INT32)
            .value("INT64", DataType::INT64)
            .value("INT8", DataType::INT8)
            .value("UINT8", DataType::UINT8)
            .value("UNKNOW", DataType::UNKNOW);


        pybind11::class_<TensorInfo>(m, "TensorInfo")
            .def(pybind11::init())
            .def_readwrite("name", &TensorInfo::name)
            .def_readwrite("shape", &TensorInfo::shape)
            .def_readwrite("dtype", &TensorInfo::dtype)
            .def("__str__", [](const TensorInfo& self) {
                return format("TensorInfo(name={}, shape={}, dtype={})",
                              self.name, vector_to_string(self.shape), datatype_to_string(self.dtype));
            })
            .def("__repr__", [](const TensorInfo& self) {
                return format("TensorInfo(name={}, shape={}, dtype={})",
                              self.name, vector_to_string(self.shape), datatype_to_string(self.dtype));
            });

        pybind11::class_<Runtime>(m, "Runtime")
            .def(pybind11::init())
            .def("init", &Runtime::init)
            .def("infer", infer_with_numpy_map)
            .def("infer", infer_with_tensor_map)
            .def("infer", infer_with_tensor_vector)
            .def("bind_input_tensor", &Runtime::bind_input_tensor)
            .def("bind_output_tensor", &Runtime::bind_output_tensor)
            .def("infer", [](Runtime& self) { self.infer(); })
            .def("get_output_tensor",
                 [](Runtime& self, const std::string& name) {
                     Tensor* output = self.get_output_tensor(name);
                     if (output == nullptr) {
                         return pybind11::cast(nullptr);
                     }
                     return pybind11::cast(*output);
                 })
            .def("num_inputs", &Runtime::num_inputs)
            .def("num_outputs", &Runtime::num_outputs)
            .def("get_input_info", &Runtime::get_input_info)
            .def("get_output_info", &Runtime::get_output_info)
            .def_readonly("option", &Runtime::option);
    }
}
