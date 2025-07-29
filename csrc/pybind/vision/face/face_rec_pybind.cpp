//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_rec/seetaface.h"

namespace modeldeploy::vision {
    void bind_face_rec(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceIDPreprocessor>(m, "SeetaFaceIDPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const face::SeetaFaceIDPreprocessor& self,
                    const std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in SeetaFaceIDPreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"))
            .def_property("size", &face::SeetaFaceIDPreprocessor::get_size,
                          &face::SeetaFaceIDPreprocessor::set_size);

        pybind11::class_<face::SeetaFaceIDPostprocessor>(m, "SeetaFaceIDPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](face::SeetaFaceIDPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<FaceRecognitionResult> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "SeetaFaceIDPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("run",
                 [](face::SeetaFaceIDPostprocessor& self,
                    std::vector<pybind11::array>& input_arrays) {
                     std::vector<FaceRecognitionResult> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_arrays, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime SeetaFaceIDPostprocessor in LprDetPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"));


        pybind11::class_<face::SeetaFaceID, BaseModel>(m, "SeetaFaceID")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceID& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     const auto image_data = ImageData::from_mat(&mat);
                     FaceRecognitionResult result;
                     self.predict(image_data, &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](face::SeetaFaceID& self,
                    const std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         const auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<FaceRecognitionResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &face::SeetaFaceID::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::SeetaFaceID::get_postprocessor);
    }
} // namespace modeldeploy
