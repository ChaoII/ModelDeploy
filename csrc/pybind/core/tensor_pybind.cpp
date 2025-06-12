//
// Created by aichao on 2025/6/12.
//


#include "csrc/pybind/utils/utils.h"

namespace modeldeploy {
    void bind_tensor(pybind11::module& m) {
        pybind11::class_<Tensor>(m, "Tensor")
            .def(pybind11::init<>(), "Default Constructor")
            .def_property("name", &Tensor::get_name, &Tensor::set_name)
            .def_property_readonly("shape", &Tensor::shape)
            .def_property_readonly("dtype", &Tensor::dtype)
            .def("to_numpy", [](const Tensor& self) { return tensor_to_pyarray(self); })
            .def("data", pybind11::overload_cast<>(&Tensor::data, pybind11::const_))
            .def("from_numpy",
                 [](Tensor& self, pybind11::array& pyarray,
                    const bool share_buffer = false) {
                     pyarray_to_tensor(pyarray, &self, share_buffer);
                 })
            .def("from_external_data",
                 [](const std::string& name, const size_t data_addr,
                    const std::vector<int64_t>& shape, const std::string& data_type) {
                     auto tensor_data_type = DataType::UNKNOW;
                     if (data_type == "FP32") {
                         tensor_data_type = DataType::FP32;
                     }
                     else if (data_type == "INT32") {
                         tensor_data_type = DataType::INT32;
                     }
                     else if (data_type == "INT64") {
                         tensor_data_type = DataType::INT64;
                     }
                     else {
                         MD_LOG_FATAL <<
                             "FDTensor.from_external_data, datatype " << data_type << " is not supported." << std::endl;
                     }
                     void* data_ptr = nullptr;
                     data_ptr = reinterpret_cast<void*>(data_addr);
                     Tensor fd_tensor;
                     fd_tensor.from_external_memory(data_ptr, shape, tensor_data_type, nullptr, name);
                     return fd_tensor;
                 })
            .def("print_info", &Tensor::print);
    }
}
