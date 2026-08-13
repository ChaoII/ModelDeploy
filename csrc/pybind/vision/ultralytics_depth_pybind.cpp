//
// Created by aichao on 2026/8/13.
//

#include "pybind/utils/utils.h"
#include "vision/depth/ultralytics_depth.h"

namespace modeldeploy::vision {
    void bind_ultralytics_depth(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsDepthPreprocessor>(m, "UltralyticsDepthPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsDepthPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData(std::move(cv_image)));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsDepthPreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"))
            .def_property("size", &detection::UltralyticsDepthPreprocessor::get_size,
                          &detection::UltralyticsDepthPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsDepthPreprocessor::get_padding_value,
                          &detection::UltralyticsDepthPreprocessor::set_padding_value);

        pybind11::class_<detection::UltralyticsDepthPostprocessor>(
                m, "UltralyticsDepthPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsDepthPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<DepthResult> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsDepthPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"));

        pybind11::class_<detection::UltralyticsDepth, BaseModel>(m, "UltralyticsDepth")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsDepth& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     DepthResult result;
                     self.predict(ImageData(std::move(mat)), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsDepth& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData(std::move(cv_image)));
                     }
                     std::vector<DepthResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsDepth::get_preprocessor)
            .def_property_readonly("postprocessor",
                                    &detection::UltralyticsDepth::get_postprocessor)
            .def("clone", [](const detection::UltralyticsDepth& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
