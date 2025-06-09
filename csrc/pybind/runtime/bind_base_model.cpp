//
// Created by aichao on 2025/6/9.
//


#include "csrc/runtime/runtime_option.h"
#include <pybind11/pybind11.h>

namespace modeldeploy {
    // void bind_ort_option(pybind11::module& m);

    void bind_runtime_option(pybind11::module& m) {
        // bind_ort_option(m);

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
    }
} // namespace modeldeploy
