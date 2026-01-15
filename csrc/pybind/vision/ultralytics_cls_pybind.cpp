//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/classification/classification.h"

namespace modeldeploy::vision {
    void bind_classification(const pybind11::module& m) {
        pybind11::class_<classification::ClassificationPreprocessor>(m, "ClassificationPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const classification::ClassificationPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in ClassificationPreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"))
            .def("enable_center_crop", &classification::ClassificationPreprocessor::enable_center_crop)
            .def("disable_center_crop", &classification::ClassificationPreprocessor::disable_center_crop)
            .def_property("size", &classification::ClassificationPreprocessor::get_size,
                          &classification::ClassificationPreprocessor::set_size);

        pybind11::class_<classification::ClassificationPostprocessor>(
                m, "ClassificationPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const classification::ClassificationPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<ClassifyResult> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in ClassificationPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("run",
                 [](const classification::ClassificationPostprocessor& self,
                    std::vector<pybind11::array>& input_array) {
                     std::vector<ClassifyResult> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "ClassificationPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("input_array"))
            .def("set_multi_label", &classification::ClassificationPostprocessor::set_multi_label);


        pybind11::class_<classification::Classification, BaseModel>(m, "Classification")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](classification::Classification& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     ClassifyResult result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](classification::Classification& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<ClassifyResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &classification::Classification::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &classification::Classification::get_postprocessor);
    }
} // namespace modeldeploy
