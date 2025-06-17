//
// Created by aichao on 2025/6/9.
//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "opencv2/opencv.hpp"
#include "csrc/core/tensor.h"
#include <type_traits>
#include <csrc/core/md_log.h>
#include "csrc/vision/common/struct.h"


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
