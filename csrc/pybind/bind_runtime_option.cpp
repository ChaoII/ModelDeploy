//
// Created by aichao on 2025/6/9.
//

#include "csrc/base_model.h"
#include <pybind11/pybind11.h>

namespace modeldeploy {
    void bind_base_model(pybind11::module& m) {
        pybind11::class_<BaseModel>(m, "BaseModel")
            .def(pybind11::init<>(), "Default Constructor")
            .def("model_name", &BaseModel::name)
            .def("num_inputs", &BaseModel::num_inputs)
            .def("num_outputs", &BaseModel::num_outputs)
            .def("get_input_info", &BaseModel::get_input_info)
            .def("get_output_info", &BaseModel::get_output_info)
            .def("get_custom_meta_data", &BaseModel::get_custom_meta_data)
            .def("initialized", &BaseModel::is_initialized)
            .def_readwrite("runtime_option", &BaseModel::runtime_option);
    }
} // namespace fastdeploy
