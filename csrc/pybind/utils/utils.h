//
// Created by aichao on 2025/6/9.
//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "opencv2/opencv.hpp"
#include "core/tensor.h"
#include <type_traits>
#include <core/md_log.h>
#include "vision/common/struct.h"
#include <filesystem>
#include <Python.h>


namespace modeldeploy {
    pybind11::dtype md_data_type_to_numpy_data_type(const DataType& md_dtype);

    DataType numpy_data_type_to_md_data_type(const pybind11::dtype& np_dtype);

    void pyarray_to_tensor(pybind11::array& pyarray, Tensor* tensor, bool share_buffer = false);

    void pyarray_to_tensor_list(std::vector<pybind11::array>& pyarray,
                                std::vector<Tensor>* tensor,
                                bool share_buffer = false);

    pybind11::array tensor_to_pyarray(const Tensor& tensor);


#ifdef BUILD_VISION
    cv::Mat pyarray_to_cv_mat(const pybind11::array& pyarray);

    pybind11::array cv_mat_to_pyarray(const cv::Mat& mat);

    vision::Point2f pyarray_to_point2f(pybind11::array& pyarray);

    vision::Point3f pyarray_to_point3f(pybind11::array& pyarray);

    vision::Rect2f pyarray_to_rect2f(const pybind11::array& pyarray);

    vision::RotatedRect pyarray_to_rotated_rect(pybind11::array& pyarray);


#endif

    template <typename T>
    DataType c_type_to_md_data_type() {
        if (std::is_same_v<T, int32_t>) {
            return DataType::INT32;
        }
        if (std::is_same_v<T, int64_t>) {
            return DataType::INT64;
        }
        if (std::is_same_v<T, float>) {
            return DataType::FP32;
        }
        if (std::is_same_v<T, double>) {
            return DataType::FP64;
        }
        if (std::is_same_v<T, int8_t>) {
            return DataType::INT8;
        }
        MD_LOG_FATAL << "CTypeToFDDataType only support int8/int32/int64/float32/float64 now." << std::endl;
        return DataType::FP32;
    }


} // namespace modeldeploy

// pybind11/detail 命名空间内定义 std::filesystem::path 的 type caster，
// 使绑定形参 std::filesystem::path 同时接受 str/bytes/os.PathLike(pathlib.Path)。
// 放在 modeldeploy 命名空间闭合之后。
PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
template <>
struct type_caster<std::filesystem::path> {
public:
    PYBIND11_TYPE_CASTER(std::filesystem::path, const_name("os.PathLike[str]"));

    bool load(handle src, bool) {
        // PyOS_FSPath 接受 str/bytes 和任何有 __fspath__ 的对象(pathlib.Path)
        PyObject* fspath = PyOS_FSPath(src.ptr());
        if (!fspath) {
            PyErr_Clear();
            return false;
        }
        // 统一转成 UTF-8 字符串（str 直接；bytes 解码）
        Py_ssize_t size = 0;
        const char* data = nullptr;
        if (PyUnicode_Check(fspath)) {
            data = PyUnicode_AsUTF8AndSize(fspath, &size);
            if (!data) {
                Py_DECREF(fspath);
                PyErr_Clear();
                return false;
            }
        } else if (PyBytes_Check(fspath)) {
            // 3.13 移除 2 参 getter（PyBytes_AsStringAndSize(obj,&size)），
            // 改用所有版本均有的 3 参形式：成功返回 0 并填充 buffer/size。
            char* buffer = nullptr;
            if (PyBytes_AsStringAndSize(fspath, &buffer, &size) != 0) {
                Py_DECREF(fspath);
                PyErr_Clear();
                return false;
            }
            data = buffer;
        } else {
            Py_DECREF(fspath);
            return false;
        }
        value = std::filesystem::path(std::string(data, size));
        Py_DECREF(fspath);
        return true;
    }

    static handle cast(const std::filesystem::path& src, return_value_policy, handle) {
        return PyUnicode_FromString(src.string().c_str());
    }
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
