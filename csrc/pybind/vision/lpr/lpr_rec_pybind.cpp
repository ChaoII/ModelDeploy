//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/lpr/lpr_rec/lpr_rec.h"

namespace modeldeploy::vision {
    void bind_lpr_rec(const pybind11::module& m) {
        pybind11::class_<lpr::LprRecPreprocessor>(m, "LprRecPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const lpr::LprRecPreprocessor& self,
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
                             "Failed to preprocess the input data in UltralyticsPosePreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"))
            .def_property("size", &lpr::LprRecPreprocessor::get_size,
                          &lpr::LprRecPreprocessor::set_size);


        pybind11::class_<lpr::LprRecPostprocessor>(m, "LprRecPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const lpr::LprRecPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<LprResult> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "LprDetPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("run",
                 [](const lpr::LprRecPostprocessor& self,
                    std::vector<pybind11::array>& input_array) {
                     std::vector<LprResult> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "LprDetPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"));

        pybind11::class_<lpr::LprRecognizer, BaseModel>(m, "LprRecognizer")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](lpr::LprRecognizer& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     LprResult result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](lpr::LprRecognizer& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<LprResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &lpr::LprRecognizer::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &lpr::LprRecognizer::get_postprocessor);
    }
} // namespace modeldeploy
