//
// Created by aichao on 2025/6/9.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/detection/preprocessor.h"
#include "csrc/vision/detection/postprocessor.h"
#include "csrc/vision/detection/ultralytics_det.h"

namespace modeldeploy::vision {
    void bind_ultralytics_det(pybind11::module& m) {
        pybind11::class_<detection::UltralyticsPreprocessor>(m, "UltralyticsPreprocessor")
            .def(pybind11::init<>())
            .def(
                "run",
                [](const vision::detection::UltralyticsPreprocessor& self,
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
                            "Failed to preprocess the input data in YOLOv8Preprocessor.");
                    }
                    return make_pair(outputs, records);
                })
            .def_property("size", &vision::detection::UltralyticsPreprocessor::get_size,
                          &vision::detection::UltralyticsPreprocessor::set_size)
            .def_property("padding_value",
                          &vision::detection::UltralyticsPreprocessor::get_padding_value,
                          &vision::detection::UltralyticsPreprocessor::set_padding_value)
            .def_property("is_scale_up",
                          &vision::detection::UltralyticsPreprocessor::get_scale_up,
                          &vision::detection::UltralyticsPreprocessor::set_scale_up)
            .def_property("is_mini_pad",
                          &vision::detection::UltralyticsPreprocessor::get_mini_pad,
                          &vision::detection::UltralyticsPreprocessor::set_mini_pad)
            .def_property("stride", &vision::detection::UltralyticsPreprocessor::get_stride,
                          &vision::detection::UltralyticsPreprocessor::set_stride);

        pybind11::class_<vision::detection::UltralyticsPostprocessor>(
                m, "YOLOv8Postprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const vision::detection::UltralyticsPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "YOLOv8Postprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](vision::detection::UltralyticsPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "YOLOv8Postprocessor.");
                     }
                     return results;
                 })
            .def_property("conf_threshold",
                          &vision::detection::UltralyticsPostprocessor::get_conf_threshold,
                          &vision::detection::UltralyticsPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &vision::detection::UltralyticsPostprocessor::get_nms_threshold,
                          &vision::detection::UltralyticsPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsDet, BaseModel>(m, "UltralyticsDet")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsDet& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<DetectionResult> res;
                     self.predict(mat, &res);
                     return res;
                 })
            .def("batch_predict",
                 [](detection::UltralyticsDet& self,
                    std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     images.reserve(data.size());
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<std::vector<vision::DetectionResult>> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property_readonly("preprocessor",
                                   &vision::detection::UltralyticsDet::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &vision::detection::UltralyticsDet::get_postprocessor);
    }
} // namespace fastdeploy
