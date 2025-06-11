//
// Created by aichao on 2025/6/9.
//


#include "csrc/runtime/runtime_option.h"
#include <pybind11/pybind11.h>

namespace modeldeploy {
    void bind_runtime_option(pybind11::module& m) {
        pybind11::enum_<Device>(m, "Device")
            .value("CPU", CPU)
            .value("GPU", GPU)
            .export_values();
        pybind11::enum_<Backend>(m, "Backend")
            .value("NONE", NONE)
            .value("ORT", ORT)
            .export_values();

        pybind11::class_<RuntimeOption>(m, "RuntimeOption")
            .def(pybind11::init())
            .def("set_model_path", &RuntimeOption::set_model_path)

            .def("use_gpu", &RuntimeOption::use_gpu)
            .def("use_cpu", &RuntimeOption::use_cpu)
            .def("set_ort_graph_opt_level", &RuntimeOption::set_ort_graph_opt_level)
            .def_readwrite("model_buffer", &RuntimeOption::model_buffer)
            .def_readwrite("ort_option", &RuntimeOption::ort_option)
            .def("set_external_stream", &RuntimeOption::set_external_stream)
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

        pybind11::class_<OrtBackendOption>(m, "OrtBackendOption")
            .def(pybind11::init())
            .def_readwrite("model_from_memory", &OrtBackendOption::model_from_memory)
            .def_readwrite("graph_optimization_level", &OrtBackendOption::graph_optimization_level)
            .def_readwrite("intra_op_num_threads", &OrtBackendOption::intra_op_num_threads)
            .def_readwrite("inter_op_num_threads", &OrtBackendOption::inter_op_num_threads)
            .def_readwrite("execution_mode", &OrtBackendOption::execution_mode)
            .def_readwrite("device", &OrtBackendOption::device)
            .def_readwrite("device_id", &OrtBackendOption::device_id)
            .def_readwrite("external_stream", &OrtBackendOption::external_stream_)
            .def_readwrite("enable_fp16", &OrtBackendOption::enable_fp16)
            .def_readwrite("optimized_model_filepath", &OrtBackendOption::optimized_model_filepath)
            .def_readwrite("model_buffer", &OrtBackendOption::model_buffer)
            .def_readwrite("model_filepath", &OrtBackendOption::model_filepath)
            .def_readwrite("enable_trt", &OrtBackendOption::enable_trt)
            .def_readwrite("trt_min_shape", &OrtBackendOption::trt_min_shape)
            .def_readwrite("trt_opt_shape", &OrtBackendOption::trt_opt_shape)
            .def_readwrite("trt_max_shape", &OrtBackendOption::trt_max_shape)
            .def("set_cpu_thread_num", &OrtBackendOption::set_cpu_thread_num);
    }
} // namespace modeldeploy
