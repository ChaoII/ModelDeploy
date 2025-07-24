//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/obb/ultralytics_obb.h"

namespace modeldeploy::vision {
    void bind_ultralytics_obb(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsObbPreprocessor>(m, "UltralyticsObbPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsObbPreprocessor& self,
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
                             "Failed to preprocess the input data in UltralyticsObbPreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"))
            .def_property("size", &detection::UltralyticsObbPreprocessor::get_size,
                          &detection::UltralyticsObbPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsObbPreprocessor::get_padding_value,
                          &detection::UltralyticsObbPreprocessor::set_padding_value);

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
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
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
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def_property("conf_threshold",
                          &detection::UltralyticsObbPostprocessor::get_conf_threshold,
                          &detection::UltralyticsObbPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsObbPostprocessor::get_nms_threshold,
                          &detection::UltralyticsObbPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsObb, BaseModel>(m, "UltralyticsObb")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsObb& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<ObbResult> result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsObb& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<std::vector<ObbResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsObb::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &detection::UltralyticsObb::get_postprocessor);
    }
} // namespace modeldeploy
