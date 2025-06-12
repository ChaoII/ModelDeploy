//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_age/seetaface_age.h"

namespace modeldeploy::vision {
    void bind_face_age(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAgePreprocessor>(m, "SeetaFaceAgePreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const face::SeetaFaceAgePreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in SeetaFaceAgePreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"))
            .def_property("size", &face::SeetaFaceAgePreprocessor::get_size,
                          &face::SeetaFaceAgePreprocessor::set_size);

        pybind11::class_<face::SeetaFaceAgePostprocessor>(m, "SeetaFaceAgePostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const face::SeetaFaceAgePostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<int> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "SeetaFaceAgePostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("run",
                 [](face::SeetaFaceAgePostprocessor& self,
                    std::vector<pybind11::array>& input_arrays) {
                     std::vector<int> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_arrays, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime SeetaFaceAgePostprocessor in LprDetPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"));


        pybind11::class_<face::SeetaFaceAge, BaseModel>(m, "SeetaFaceAge")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceAge& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     int result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](face::SeetaFaceAge& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<cv::Mat> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         _images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<int> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &face::SeetaFaceAge::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::SeetaFaceAge::get_postprocessor);
    }
} // namespace modeldeploy
