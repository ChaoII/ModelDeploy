//
// Created by aichao on 2025/6/9.
//

#include "pybind/utils/utils.h"

namespace modeldeploy {
    pybind11::dtype md_data_type_to_numpy_data_type(const DataType& md_dtype) {
        pybind11::dtype dt;
        if (md_dtype == DataType::INT32) {
            dt = pybind11::dtype::of<int32_t>();
        }
        else if (md_dtype == DataType::INT64) {
            dt = pybind11::dtype::of<int64_t>();
        }
        else if (md_dtype == DataType::FP32) {
            dt = pybind11::dtype::of<float>();
        }
        else if (md_dtype == DataType::FP64) {
            dt = pybind11::dtype::of<double>();
        }
        else if (md_dtype == DataType::UINT8) {
            dt = pybind11::dtype::of<uint8_t>();
        }
        else if (md_dtype == DataType::INT8) {
            dt = pybind11::dtype::of<int8_t>();
        }
        else {
            MD_LOG_FATAL << "The function doesn't support data type of %s." << datatype_to_string(md_dtype) <<
                std::endl;
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

    void pyarray_to_tensor(pybind11::array& pyarray, Tensor* tensor, const bool share_buffer) {
        const auto dtype = numpy_data_type_to_md_data_type(pyarray.dtype());
        std::vector<int64_t> data_shape;
        data_shape.insert(data_shape.begin(), pyarray.shape(), pyarray.shape() + pyarray.ndim());
        if (share_buffer) {
            tensor->from_external_memory(pyarray.mutable_data(), data_shape, dtype);
        }
        else {
            tensor->allocate(data_shape, dtype);
            memcpy(tensor->data(), pyarray.mutable_data(), pyarray.nbytes());
        }
    }

    void pyarray_to_tensor_list(std::vector<pybind11::array>& pyarray,
                                std::vector<Tensor>* tensor,
                                const bool share_buffer) {
        tensor->resize(pyarray.size());
        for (auto i = 0; i < pyarray.size(); ++i) {
            pyarray_to_tensor(pyarray[i], &(*tensor)[i], share_buffer);
        }
    }

    pybind11::array tensor_to_pyarray(const Tensor& tensor) {
        const auto numpy_dtype = md_data_type_to_numpy_data_type(tensor.dtype());
        auto out = pybind11::array(numpy_dtype, tensor.shape());
        memcpy(out.mutable_data(), tensor.data(), tensor.byte_size());
        return out;
    }


#ifdef BUILD_VISION

    vision::Point2f pyarray_to_point2f(const pybind11::array& pyarray) {
        if (pyarray.ndim() != 1 || pyarray.shape(0) != 2)
            throw std::runtime_error("Expected a 1D numpy array of size 2 for Point2f");

        const auto ptr = pyarray.unchecked<float>(); // No bounds checking

        return vision::Point2f{ptr(0), ptr(1)};
    }

    vision::Point3f pyarray_to_point3f(const pybind11::array& pyarray) {
        if (pyarray.ndim() != 1 || pyarray.shape(0) != 3)
            throw std::runtime_error("Expected a 1D numpy array of size 3 for Point3f");

        const auto ptr = pyarray.unchecked<float>();
        return vision::Point3f{ptr(0), ptr(1), ptr(2)};
    }

    vision::Rect2f pyarray_to_rect2f(const pybind11::array& pyarray) {
        if (pyarray.ndim() != 1 || pyarray.shape(0) != 4)
            throw std::runtime_error("Expected a 1D numpy array of size 4 for Rect2f");

        const auto ptr = pyarray.unchecked<float>();
        return vision::Rect2f{ptr(0), ptr(1), ptr(2), ptr(3)};
    }

    vision::RotatedRect pyarray_to_rotated_rect(const pybind11::array& pyarray) {
        if (pyarray.ndim() != 1 || pyarray.shape(0) != 5)
            throw std::runtime_error("Expected a 1D numpy array of size 5 for RotatedRect");

        auto ptr = pyarray.unchecked<float>();
        return vision::RotatedRect{ptr(0), ptr(1), ptr(2), ptr(3), ptr(4)};
    }


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

    int numpy_data_type_to_open_cv_type_v2(const pybind11::array& pyarray) {
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

    pybind11::dtype cv_data_type_to_numpy_dtype(const int cv_depth) {
        switch (cv_depth) {
        case CV_8U:
            return pybind11::dtype::of<std::uint8_t>();
        case CV_8S:
            return pybind11::dtype::of<std::int8_t>();
        case CV_16U:
            return pybind11::dtype::of<std::uint16_t>();
        case CV_16S:
            return pybind11::dtype::of<std::int16_t>();
        case CV_32S:
            return pybind11::dtype::of<std::int32_t>();
        case CV_32F:
            return pybind11::dtype::of<float>();
        case CV_64F:
            return pybind11::dtype::of<double>();
        default:
            throw std::runtime_error("Unsupported OpenCV data type in cv_data_type_to_numpy_dtype()");
        }
    }

    cv::Mat pyarray_to_cv_mat(const pybind11::array& pyarray) {
        if (pyarray.ndim() != 3) {
            throw std::runtime_error("Expected 3D array (HWC) for image input");
        }
        const auto _cv_type = numpy_data_type_to_open_cv_type_v2(pyarray);
        const int height = pyarray.shape()[0];      //H
        const int width = pyarray.shape()[1];       //W
        const int channels = pyarray.shape()[2];    //C
        const auto cv_type = CV_MAKETYPE(_cv_type, channels);
        // 注意：pyarray::data() 返回的是 const void*
        const auto data_ptr = const_cast<void*>(pyarray.data());
        cv::Mat mat(height, width, cv_type, data_ptr);
        return mat;
    }

    pybind11::array cv_mat_to_pyarray(const cv::Mat& mat) {
        // 获取 numpy 类型（如 np.uint8）
        auto np_dtype = cv_data_type_to_numpy_dtype(mat.depth());

        // 构造 shape 和 strides（以 HWC 格式）
        const std::vector<int64_t> shape = {mat.rows, mat.cols, mat.channels()};
        const std::vector strides = {
            static_cast<size_t>(mat.step[0]),
            static_cast<size_t>(mat.step[1]), (mat.elemSize1())
        };
        // 构造共享内存 numpy array，不复制数据
        return pybind11::array{pybind11::dtype(np_dtype), shape, strides, mat.data};
    }

#endif
}
