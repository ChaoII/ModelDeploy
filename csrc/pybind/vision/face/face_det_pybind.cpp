//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_det/scrfd.h"

namespace modeldeploy::vision {
    void bind_face_det(const pybind11::module& m) {
        pybind11::class_<face::ScrfdPreprocessor>(m, "ScrfdPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const face::ScrfdPreprocessor& self,
                    const std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         const auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in ScrfdPreprocessor.");
                     }
                     return make_pair(outputs, records);
                 }, pybind11::arg("im_list"))
            .def_property("size", &face::ScrfdPreprocessor::get_size,
                          &face::ScrfdPreprocessor::set_size)
            .def_property("padding_value",
                          &face::ScrfdPreprocessor::get_padding_value,
                          &face::ScrfdPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &face::ScrfdPreprocessor::get_scale_up,
                          &face::ScrfdPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &face::ScrfdPreprocessor::get_mini_pad,
                          &face::ScrfdPreprocessor::set_mini_pad)
            .def_property("stride", &face::ScrfdPreprocessor::get_stride,
                          &face::ScrfdPreprocessor::set_stride);

        pybind11::class_<face::ScrfdPostprocessor>(m, "ScrfdPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](face::ScrfdPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<KeyPointsResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "ScrfdPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def("run",
                 [](face::ScrfdPostprocessor& self,
                    std::vector<pybind11::array>& input_arrays,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<KeyPointsResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_arrays, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime ScrfdPostprocessor in LprDetPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def_property("conf_threshold",
                          &face::ScrfdPostprocessor::get_conf_threshold,
                          &face::ScrfdPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &face::ScrfdPostprocessor::get_nms_threshold,
                          &face::ScrfdPostprocessor::set_nms_threshold)
            .def_property("landmarks_per_face_",
                          &face::ScrfdPostprocessor::get_landmarks_per_face,
                          &face::ScrfdPostprocessor::set_landmarks_per_face);

        pybind11::class_<face::Scrfd, BaseModel>(m, "Scrfd")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::Scrfd& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<KeyPointsResult> result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](face::Scrfd& self,
                    const std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         const auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<std::vector<KeyPointsResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &face::Scrfd::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::Scrfd::get_postprocessor);
    }
} // namespace modeldeploy
