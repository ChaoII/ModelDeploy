//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_rec/seetaface.h"

namespace modeldeploy::vision {
    void bind_face_rec(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceIDPreprocessor>(m, "SeetaFaceIDPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const face::SeetaFaceIDPreprocessor& self,
                   std::vector<pybind11::array>& im_list) {
                    std::vector<cv::Mat> images;
                    images.reserve(im_list.size());
                    for (auto& image : im_list) {
                        images.push_back(pyarray_to_cv_mat(image));
                    }
                    std::vector<Tensor> outputs;
                    if (!self.run(&images, &outputs)) {
                        throw std::runtime_error(
                            "Failed to preprocess the input data in SeetaFaceIDPreprocessor.");
                    }
                    pybind11::array array;
                    tensor_list_to_pyarray(outputs, array);
                    return array;
                })
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
                 })
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
                 });


        pybind11::class_<face::SeetaFaceID, BaseModel>(m, "SeetaFaceID")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceID& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     FaceRecognitionResult res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](face::SeetaFaceID& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<FaceRecognitionResult> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &face::SeetaFaceID::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::SeetaFaceID::get_postprocessor);
    }
} // namespace modeldeploy
