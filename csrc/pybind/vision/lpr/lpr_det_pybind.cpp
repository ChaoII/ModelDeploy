//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/lpr/lpr_det/lpr_det.h"

namespace modeldeploy::vision {
    void bind_lpr_det(const pybind11::module& m) {
        pybind11::class_<lpr::LprDetPreprocessor>(m, "LprDetPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const lpr::LprDetPreprocessor& self,
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
                            "Failed to preprocess the input data in LprDetPreprocessor.");
                    }
                    pybind11::array array;
                    tensor_list_to_pyarray(outputs, array);
                    return make_pair(array, records);
                })
            .def_property("size", &lpr::LprDetPreprocessor::get_size,
                          &lpr::LprDetPreprocessor::set_size)
            .def_property("padding_value",
                          &lpr::LprDetPreprocessor::get_padding_value,
                          &lpr::LprDetPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &lpr::LprDetPreprocessor::get_scale_up,
                          &lpr::LprDetPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &lpr::LprDetPreprocessor::get_mini_pad,
                          &lpr::LprDetPreprocessor::set_mini_pad)
            .def_property("stride", &lpr::LprDetPreprocessor::get_stride,
                          &lpr::LprDetPreprocessor::set_stride);

        pybind11::class_<lpr::LprDetPostprocessor>(m, "LprDetPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const lpr::LprDetPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionLandmarkResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "LprDetPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](const lpr::LprDetPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionLandmarkResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "LprDetPostprocessor.");
                     }
                     return results;
                 })
            .def_property("conf_threshold",
                          &lpr::LprDetPostprocessor::get_conf_threshold,
                          &lpr::LprDetPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &lpr::LprDetPostprocessor::get_nms_threshold,
                          &lpr::LprDetPostprocessor::set_nms_threshold)
            .def_property("landmarks_per_card_",
                          &lpr::LprDetPostprocessor::get_landmarks_per_card,
                          &lpr::LprDetPostprocessor::set_landmarks_per_card);

        pybind11::class_<lpr::LprDetection, BaseModel>(m, "LprDetection")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](lpr::LprDetection& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<DetectionLandmarkResult> res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](lpr::LprDetection& self,
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
                                   &lpr::LprDetection::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &lpr::LprDetection::get_postprocessor);
    }
} // namespace modeldeploy
