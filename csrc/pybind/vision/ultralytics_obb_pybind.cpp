//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/obb/ultralytics_obb.h"

namespace modeldeploy::vision {
    void bind_ultralytics_obb(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsObbPreprocessor>(m, "UltralyticsObbPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const detection::UltralyticsObbPreprocessor& self,
                   std::vector<pybind11::array>& im_list) {
                    std::vector<cv::Mat> images;
                    images.reserve(im_list.size());
                    for (auto& image : im_list) {
                        images.push_back(pyarray_to_cv_mat(image));
                    }
                    std::vector<LetterBoxRecord> records;
                    std::vector<Tensor> outputs;
                    if (!self.run(&images, &outputs, &records)) {
                        throw std::runtime_error(
                            "Failed to preprocess the input data in UltralyticsObbPreprocessor.");
                    }
                    std::vector<pybind11::array> arrays;
                    tensor_list_to_pyarray_list(outputs, arrays);
                    return make_pair(arrays, records);
                })
            .def_property("size", &detection::UltralyticsObbPreprocessor::get_size,
                          &detection::UltralyticsObbPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsObbPreprocessor::get_padding_value,
                          &detection::UltralyticsObbPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &detection::UltralyticsObbPreprocessor::get_scale_up,
                          &detection::UltralyticsObbPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &detection::UltralyticsObbPreprocessor::get_mini_pad,
                          &detection::UltralyticsObbPreprocessor::set_mini_pad)
            .def_property("stride", &detection::UltralyticsObbPreprocessor::get_stride,
                          &detection::UltralyticsObbPreprocessor::set_stride);

        pybind11::class_<detection::UltralyticsObbPostprocessor>(
                m, "UltralyticsObbPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsObbPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<ObbResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsObbPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](detection::UltralyticsObbPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<ObbResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsObbPostprocessor.");
                     }
                     return results;
                 })
            .def_property("conf_threshold",
                          &detection::UltralyticsObbPostprocessor::get_conf_threshold,
                          &detection::UltralyticsObbPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsObbPostprocessor::get_nms_threshold,
                          &detection::UltralyticsObbPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsObb, BaseModel>(m, "UltralyticsObb")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsObb& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<ObbResult> res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](detection::UltralyticsObb& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<std::vector<ObbResult>> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsObb::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &detection::UltralyticsObb::get_postprocessor);
    }
} // namespace fastdeploy
