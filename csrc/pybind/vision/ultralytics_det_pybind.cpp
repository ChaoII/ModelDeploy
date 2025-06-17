//
// Created by aichao on 2025/6/9.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/detection/ultralytics_det.h"

namespace modeldeploy::vision {
    void bind_ultralytics_det(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsPreprocessor>(m, "UltralyticsPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPreprocessor& self,
                    const std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsPreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"), pybind11::return_value_policy::move)
            .def_property("size", &detection::UltralyticsPreprocessor::get_size,
                          &detection::UltralyticsPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsPreprocessor::get_padding_value,
                          &detection::UltralyticsPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &detection::UltralyticsPreprocessor::get_scale_up,
                          &detection::UltralyticsPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &detection::UltralyticsPreprocessor::get_mini_pad,
                          &detection::UltralyticsPreprocessor::set_mini_pad)
            .def_property("stride", &detection::UltralyticsPreprocessor::get_stride,
                          &detection::UltralyticsPreprocessor::set_stride);

        pybind11::class_<detection::UltralyticsPostprocessor>(
                m, "UltralyticsPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in UltralyticsPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def("run",
                 [](detection::UltralyticsPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in UltralyticsPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def_property("conf_threshold",
                          &detection::UltralyticsPostprocessor::get_conf_threshold,
                          &detection::UltralyticsPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsPostprocessor::get_nms_threshold,
                          &detection::UltralyticsPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsDet, BaseModel>(m, "UltralyticsDet")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsDet& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<DetectionResult> result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsDet& self,
                    const std::vector<pybind11::array>& images) {
                     std::vector<cv::Mat> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         _images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<std::vector<DetectionResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsDet::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &detection::UltralyticsDet::get_postprocessor);
    }
} // namespace fastdeploy
