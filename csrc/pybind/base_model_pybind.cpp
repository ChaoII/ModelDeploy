//
// Created by aichao on 2025/6/9.
//

#include "csrc/base_model.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace modeldeploy {
    void bind_base_model(pybind11::module& m) {
        pybind11::class_<BaseModel>(m, "BaseModel")
            .def(pybind11::init<>(), "Default Constructor")
            .def("model_name", &BaseModel::name)
            .def("num_inputs", &BaseModel::num_inputs)
            .def("num_outputs", &BaseModel::num_outputs)
            .def("get_input_info", &BaseModel::get_input_info, pybind11::arg("index"))
            .def("get_output_info", &BaseModel::get_output_info, pybind11::arg("index"))
            .def("get_custom_meta_data",
                 [](BaseModel& self) {
                     const std::map<std::string, std::string> meta = self.get_custom_meta_data();
                     pybind11::dict py_meta;
                     for (const auto& kv : meta) {
                         py_meta[pybind11::str(kv.first)] = pybind11::str(kv.second);
                     }
                     return py_meta;
                 })
            .def("initialized", &BaseModel::is_initialized)
            .def_readwrite("runtime_option", &BaseModel::runtime_option);
    }
} // namespace fastdeploy
