//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_det/scrfd.h"

namespace modeldeploy::vision {
    void bind_face_det(const pybind11::module& m) {
        pybind11::class_<face::ScrfdPreprocessor>(m, "ScrfdPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const face::ScrfdPreprocessor& self,
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
                            "Failed to preprocess the input data in ScrfdPreprocessor.");
                    }
                    pybind11::array array;
                    tensor_list_to_pyarray(outputs, array);
                    return make_pair(array, records);
                })
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
                     std::vector<std::vector<DetectionLandmarkResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "ScrfdPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](face::ScrfdPostprocessor& self,
                    std::vector<pybind11::array>& input_arrays,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionLandmarkResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_arrays, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime ScrfdPostprocessor in LprDetPostprocessor.");
                     }
                     return results;
                 })
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
                 [](face::Scrfd& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<DetectionLandmarkResult> result;
                     self.predict(mat, &result);
                     return result;
                 })
            .def("batch_predict",
                 [](face::Scrfd& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<std::vector<DetectionLandmarkResult>> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &face::Scrfd::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::Scrfd::get_postprocessor);
    }
} // namespace modeldeploy
