//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/classification/ultralytics_cls.h"

namespace modeldeploy::vision {
    void bind_ultralytics_cls(const pybind11::module& m) {
        pybind11::class_<classification::UltralyticsClsPreprocessor>(m, "UltralyticsClsPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const classification::UltralyticsClsPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsPosePreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"))
            .def_property("size", &classification::UltralyticsClsPreprocessor::get_size,
                          &classification::UltralyticsClsPreprocessor::set_size);

        pybind11::class_<classification::UltralyticsClsPostprocessor>(
                m, "UltralyticsClsPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const classification::UltralyticsClsPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<ClassifyResult> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in UltralyticsClsPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("run",
                 [](const classification::UltralyticsClsPostprocessor& self,
                    std::vector<pybind11::array>& input_array) {
                     std::vector<ClassifyResult> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsObbPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("input_array"));


        pybind11::class_<classification::UltralyticsCls, BaseModel>(m, "UltralyticsCls")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](classification::UltralyticsCls& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     ClassifyResult result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](classification::UltralyticsCls& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<cv::Mat> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         _images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<ClassifyResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &classification::UltralyticsCls::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &classification::UltralyticsCls::get_postprocessor);
    }
} // namespace modeldeploy
