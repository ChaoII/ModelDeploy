//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_gender/seetaface_gender.h"

namespace modeldeploy::vision {
    void bind_face_gender(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceGenderPreprocessor>(m, "SeetaFaceGenderPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const face::SeetaFaceGenderPreprocessor& self,
                   std::vector<pybind11::array>& im_list) {
                    std::vector<cv::Mat> images;
                    images.reserve(im_list.size());
                    for (auto& image : im_list) {
                        images.push_back(pyarray_to_cv_mat(image));
                    }
                    std::vector<Tensor> outputs;
                    if (!self.run(&images, &outputs)) {
                        throw std::runtime_error(
                            "Failed to preprocess the input data in SeetaFaceGenderPreprocessor.");
                    }
                    std::vector<pybind11::array> arrays;
                    tensor_list_to_pyarray_list(outputs, arrays);
                    return arrays;
                })
            .def_property("size", &face::SeetaFaceGenderPreprocessor::get_size,
                          &face::SeetaFaceGenderPreprocessor::set_size);

        pybind11::class_<face::SeetaFaceGenderPostprocessor>(m, "SeetaFaceGenderPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const face::SeetaFaceGenderPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<int> results;
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "SeetaFaceGenderPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](face::SeetaFaceGenderPostprocessor& self,
                    std::vector<pybind11::array>& input_arrays) {
                     std::vector<int> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_arrays, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime SeetaFaceGenderPostprocessor in LprDetPostprocessor.");
                     }
                     return results;
                 });


        pybind11::class_<face::SeetaFaceGender, BaseModel>(m, "SeetaFaceGender")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceGender& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     int res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](face::SeetaFaceGender& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<int> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &face::SeetaFaceGender::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &face::SeetaFaceGender::get_postprocessor);
    }
} // namespace modeldeploy
