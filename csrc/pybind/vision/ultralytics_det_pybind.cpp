//
// Created by aichao on 2025/6/9.
//

#include "pybind/utils/utils.h"
#include "vision/detection/ultralytics_det.h"
#include "capi/vision/detection/detection_capi.h"
#include "capi/utils/internal/utils.h"

namespace modeldeploy::vision {
    void bind_ultralytics_det(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsPreprocessor>(m, "UltralyticsPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPreprocessor& self,
                    const std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData(std::move(cv_image)));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsPreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"), pybind11::return_value_policy::move)
            .def_property("size", &detection::UltralyticsPreprocessor::get_size,
                          &detection::UltralyticsPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsPreprocessor::get_padding_value,
                          &detection::UltralyticsPreprocessor::set_padding_value);

        pybind11::class_<detection::UltralyticsPostprocessor>(
                m, "UltralyticsPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in UltralyticsPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def("run",
                 [](detection::UltralyticsPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<std::vector<DetectionResult>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in UltralyticsPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"))
            .def_property("conf_threshold",
                          &detection::UltralyticsPostprocessor::get_conf_threshold,
                          &detection::UltralyticsPostprocessor::set_conf_threshold)
            .def_property("nms_threshold",
                          &detection::UltralyticsPostprocessor::get_nms_threshold,
                          &detection::UltralyticsPostprocessor::set_nms_threshold);

        pybind11::class_<detection::UltralyticsDet, BaseModel>(m, "UltralyticsDet")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsDet& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<DetectionResult> result;
                     self.predict(ImageData(std::move(mat)), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsDet& self,
                    const std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData(std::move(cv_image)));
                     }
                     std::vector<std::vector<DetectionResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def("predict_nv12",
                 [](detection::UltralyticsDet& self,
                    const pybind11::array_t<uint8_t,
                                           pybind11::array::c_style | pybind11::array::forcecast>& src_y,
                    const pybind11::array_t<uint8_t,
                                           pybind11::array::c_style | pybind11::array::forcecast>& src_uv,
                    int width, int height, int step_y, int step_uv, int src_device) {
                     if (width <= 0 || height <= 0) {
                         throw std::invalid_argument(
                             "predict_nv12: width and height must be positive.");
                     }
                     const auto y_buf = src_y.request();
                     const auto uv_buf = src_uv.request();
                     const int step_y_eff = step_y > 0 ? step_y : width;
                     const int step_uv_eff = step_uv > 0 ? step_uv : width;
                     const auto y_required = static_cast<pybind11::ssize_t>(step_y_eff) * height;
                     const auto uv_required = static_cast<pybind11::ssize_t>(step_uv_eff) * (height / 2);
                     if (y_buf.size < y_required) {
                         throw std::invalid_argument(
                             "predict_nv12: src_y buffer is too small, need " +
                             std::to_string(y_required) + " bytes but got " +
                             std::to_string(y_buf.size) + ".");
                     }
                     if (uv_buf.size < uv_required) {
                         throw std::invalid_argument(
                             "predict_nv12: src_uv buffer is too small, need " +
                             std::to_string(uv_required) + " bytes but got " +
                             std::to_string(uv_buf.size) + ".");
                     }
                     const auto* y_ptr = static_cast<const unsigned char*>(y_buf.ptr);
                     const auto* uv_ptr = static_cast<const unsigned char*>(uv_buf.ptr);

                     MDModel model{};
                     model.type = MDModelType::Detection;
                     model.format = MDModelFormat::ONNX;
                     model.model_name = nullptr;
                     model.model_content = &self;

                     MDDetectionResults c_results{};
                     const auto status = md_detection_predict_nv12(
                         &model, y_ptr, uv_ptr, width, height, step_y, step_uv,
                         static_cast<MDDevice>(src_device), &c_results);
                     if (status != MDStatusCode::Success) {
                         throw std::runtime_error(
                             "predict_nv12: md_detection_predict_nv12 failed with status=" +
                             std::to_string(status));
                     }
                     std::vector<DetectionResult> results;
                     c_results_2_detection_results(&c_results, &results);
                     md_free_detection_result(&c_results);
                     return results;
                 },
                 pybind11::arg("src_y"), pybind11::arg("src_uv"),
                 pybind11::arg("width"), pybind11::arg("height"),
                 pybind11::arg("step_y") = 0,
                 pybind11::arg("step_uv") = 0,
                 pybind11::arg("src_device") = static_cast<int>(MD_DEVICE_CPU))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsDet::get_preprocessor)
            .def_property_readonly("postprocessor",
                                    &detection::UltralyticsDet::get_postprocessor)
            .def("clone", [](const detection::UltralyticsDet& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
