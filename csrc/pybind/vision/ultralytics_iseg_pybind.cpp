//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/iseg/ultralytics_seg.h"

namespace modeldeploy::vision {
    void bind_ultralytics_iseg(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsSegPreprocessor>(m, "UltralyticsSegPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const detection::UltralyticsSegPreprocessor& self,
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
                            "Failed to preprocess the input data in YOLOv8Preprocessor.");
                    }
                    std::vector<pybind11::array> arrays;
                    tensor_list_to_pyarray_list(outputs, arrays);
                    return make_pair(arrays, records);
                })
            .def_property("size", &detection::UltralyticsSegPreprocessor::get_size,
                          &detection::UltralyticsSegPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsSegPreprocessor::get_padding_value,
                          &detection::UltralyticsSegPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &detection::UltralyticsSegPreprocessor::get_scale_up,
                          &detection::UltralyticsSegPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &detection::UltralyticsSegPreprocessor::get_mini_pad,
                          &detection::UltralyticsSegPreprocessor::set_mini_pad)
            .def_property("stride", &detection::UltralyticsSegPreprocessor::get_stride,
                          &detection::UltralyticsSegPreprocessor::set_stride);

        pybind11::class_<detection::UltralyticsSegPostprocessor>(
                m, "UltralyticsSegPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsSegPostprocessor& self,
                    std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<InstanceSegResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsSegPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](detection::UltralyticsSegPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<InstanceSegResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsSegPostprocessor.");
                     }
                     return results;
                 })
            .def_property("conf_threshold",
                          &detection::UltralyticsSegPostprocessor::get_conf_threshold,
                          &detection::UltralyticsSegPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsSegPostprocessor::get_nms_threshold,
                          &detection::UltralyticsSegPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsSeg, BaseModel>(m, "UltralyticsSeg")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsSeg& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<InstanceSegResult> res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](detection::UltralyticsSeg& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<std::vector<InstanceSegResult>> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsSeg::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &detection::UltralyticsSeg::get_postprocessor);
    }
} // namespace fastdeploy
