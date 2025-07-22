//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/pose/ultralytics_pose.h"

namespace modeldeploy::vision {
    void bind_ultralytics_pose(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsPosePreprocessor>(m, "UltralyticsPosePreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPosePreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsPosePreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"))
            .def_property("size", &detection::UltralyticsPosePreprocessor::get_size,
                          &detection::UltralyticsPosePreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsPosePreprocessor::get_padding_value,
                          &detection::UltralyticsPosePreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &detection::UltralyticsPosePreprocessor::get_scale_up,
                          &detection::UltralyticsPosePreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &detection::UltralyticsPosePreprocessor::get_mini_pad,
                          &detection::UltralyticsPosePreprocessor::set_mini_pad)
            .def_property("stride", &detection::UltralyticsPosePreprocessor::get_stride,
                          &detection::UltralyticsPosePreprocessor::set_stride);

        pybind11::class_<detection::UltralyticsPosePostprocessor>(
                m, "UltralyticsPosePostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPosePostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<PoseResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsPosePostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def("run",
                 [](detection::UltralyticsPosePostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<PoseResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsObbPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def_property("conf_threshold",
                          &detection::UltralyticsPosePostprocessor::get_conf_threshold,
                          &detection::UltralyticsPosePostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsPosePostprocessor::get_nms_threshold,
                          &detection::UltralyticsPosePostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsPose, BaseModel>(m, "UltralyticsPose")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsPose& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<PoseResult> result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsPose& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<std::vector<PoseResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsPose::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &detection::UltralyticsPose::get_postprocessor);
    }
} // namespace modeldeploy
