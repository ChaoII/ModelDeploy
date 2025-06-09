//
// Created by aichao on 2025/6/9.
//

#include "csrc/pybind/utils/utils.h"

namespace modeldeploy {
    pybind11::dtype md_data_type_to_numpy_data_type(const DataType& fd_dtype) {
        pybind11::dtype dt;
        if (fd_dtype == DataType::INT32) {
            dt = pybind11::dtype::of<int32_t>();
        }
        else if (fd_dtype == DataType::INT64) {
            dt = pybind11::dtype::of<int64_t>();
        }
        else if (fd_dtype == DataType::FP32) {
            dt = pybind11::dtype::of<float>();
        }
        else if (fd_dtype == DataType::FP64) {
            dt = pybind11::dtype::of<double>();
        }
        else if (fd_dtype == DataType::UINT8) {
            dt = pybind11::dtype::of<uint8_t>();
        }
        else if (fd_dtype == DataType::INT8) {
            dt = pybind11::dtype::of<int8_t>();
        }
        else {
            MD_LOG_FATAL << "The function doesn't support data type of %s." << to_string(fd_dtype) << std::endl;
        }
        return dt;
    }

    DataType numpy_data_type_to_md_data_type(const pybind11::dtype& np_dtype) {
        if (np_dtype.is(pybind11::dtype::of<int32_t>())) {
            return DataType::INT32;
        }
        if (np_dtype.is(pybind11::dtype::of<int64_t>())) {
            return DataType::INT64;
        }
        if (np_dtype.is(pybind11::dtype::of<float>())) {
            return DataType::FP32;
        }
        if (np_dtype.is(pybind11::dtype::of<double>())) {
            return DataType::FP64;
        }
        if (np_dtype.is(pybind11::dtype::of<uint8_t>())) {
            return DataType::UINT8;
        }
        if (np_dtype.is(pybind11::dtype::of<int8_t>())) {
            return DataType::INT8;
        }
        MD_LOG_FATAL
            << "numpy_data_type_to_md_data_type() only support int8/int32/int64/float32/float64 now." << std::endl;
        return DataType::FP32;
    }

    void py_array_to_tensor(pybind11::array& pyarray, Tensor* tensor, bool share_buffer) {
        const auto dtype = numpy_data_type_to_md_data_type(pyarray.dtype());
        std::vector<int64_t> data_shape;
        data_shape.insert(data_shape.begin(), pyarray.shape(),
                          pyarray.shape() + pyarray.ndim());
        if (share_buffer) {
            tensor->from_external_memory(pyarray.mutable_data(), data_shape, dtype);
        }
        else {
            tensor->allocate(data_shape, dtype);
            memcpy(tensor->data(), pyarray.mutable_data(), pyarray.nbytes());
        }
    }

    void pyarray_to_tensor_list(std::vector<pybind11::array>& pyarrays,
                                std::vector<Tensor>* tensors,
                                const bool share_buffer) {
        tensors->resize(pyarrays.size());
        for (auto i = 0; i < pyarrays.size(); ++i) {
            py_array_to_tensor(pyarrays[i], &(*tensors)[i], share_buffer);
        }
    }

    pybind11::array tensor_to_py_array(const Tensor& tensor) {
        const auto numpy_dtype = md_data_type_to_numpy_data_type(tensor.dtype());
        auto out = pybind11::array(numpy_dtype, tensor.shape());
        memcpy(out.mutable_data(), tensor.data(), tensor.byte_size());
        return out;
    }

#ifdef BUILD_VISION
    int numpy_data_type_to_open_cv_type(const pybind11::dtype& np_dtype) {
        if (np_dtype.is(pybind11::dtype::of<int32_t>())) {
            return CV_32S;
        }
        if (np_dtype.is(pybind11::dtype::of<int8_t>())) {
            return CV_8S;
        }
        if (np_dtype.is(pybind11::dtype::of<uint8_t>())) {
            return CV_8U;
        }
        if (np_dtype.is(pybind11::dtype::of<float>())) {
            return CV_32F;
        }
        MD_LOG_FATAL <<
            "numpy_data_type_to_open_cv_type() only support int32/int8/uint8/float32 now." << std::endl;
        return CV_8U;
    }

    int numpy_data_type_to_open_cv_type_v2(pybind11::array& pyarray) {
        if (pybind11::isinstance<pybind11::array_t<std::int32_t>>(pyarray)) {
            return CV_32S;
        }
        if (pybind11::isinstance<pybind11::array_t<std::int8_t>>(pyarray)) {
            return CV_8S;
        }
        if (pybind11::isinstance<pybind11::array_t<std::uint8_t>>(pyarray)) {
            return CV_8U;
        }
        if (pybind11::isinstance<pybind11::array_t<std::float_t>>(pyarray)) {
            return CV_32F;
        }
        MD_LOG_FATAL <<
            "numpy_data_type_to_open_cv_type_v2() only support int32/int8/uint8/float32 now." << std::endl;
        return CV_8U;
    }

    cv::Mat pyarray_to_cv_mat(pybind11::array& pyarray) {
        // auto cv_type = numpy_data_type_to_open_cv_type(pyarray.dtype());
        const auto cv_type = numpy_data_type_to_open_cv_type_v2(pyarray);
        if (pyarray.ndim() != 3) {
            MD_LOG_FATAL << "Require rank of array to be 3 with HWC format while converting it to cv::Mat." <<
                std::endl;
        }
        const int channel = pyarray.shape()[2];
        const int height = pyarray.shape()[0];
        const int width = pyarray.shape()[1];
        return cv::Mat{height, width, CV_MAKETYPE(cv_type, channel), pyarray.mutable_data()};
    }
#endif
}
