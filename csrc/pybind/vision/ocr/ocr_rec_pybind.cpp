//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/ocr/ppocr.h"

namespace modeldeploy::vision {
    void bind_ocr_rec(const pybind11::module& m) {
        // Recognizer
        pybind11::class_<ocr::RecognizerPreprocessor>(m, "RecognizerPreprocessor")
            .def(pybind11::init<>())
            .def_property("static_shape_infer",
                          &ocr::RecognizerPreprocessor::get_static_shape_infer,
                          &ocr::RecognizerPreprocessor::set_static_shape_infer)
            .def_property("rec_image_shape",
                          &ocr::RecognizerPreprocessor::get_rec_image_shape,
                          &ocr::RecognizerPreprocessor::set_rec_image_shape)
            .def("set_normalize",
                 [](ocr::RecognizerPreprocessor& self,
                    const std::vector<float>& mean, const std::vector<float>& std,
                    const bool is_scale) {
                     self.set_normalize(mean, std, is_scale);
                 })
            .def("run",
                 [](ocr::RecognizerPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     for (auto& image : im_list) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.apply(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "RecognizerPreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"));


        pybind11::class_<ocr::RecognizerPostprocessor>(
                m, "RecognizerPostprocessor")
            .def(pybind11::init<std::string>())
            .def("run",
                 [](ocr::RecognizerPostprocessor& self,
                    const std::vector<Tensor>& inputs) {
                     std::vector<std::string> texts;
                     std::vector<float> rec_scores;
                     if (!self.run(inputs, &texts, &rec_scores)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "RecognizerPostprocessor.");
                     }
                     return std::make_pair(texts, rec_scores);
                 }, pybind11::arg("inputs"))
            .def("run", [](ocr::RecognizerPostprocessor& self,
                           std::vector<pybind11::array>& input_array) {
                std::vector<Tensor> inputs;
                pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                std::vector<std::string> texts;
                std::vector<float> rec_scores;
                if (!self.run(inputs, &texts, &rec_scores)) {
                    throw std::runtime_error(
                        "Failed to preprocess the input data in "
                        "RecognizerPostprocessor.");
                }
                return std::make_pair(texts, rec_scores);
            }, pybind11::arg("inputs"));

        pybind11::class_<ocr::Recognizer, BaseModel>(m, "Recognizer")
            .def(pybind11::init<std::string, std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::Recognizer::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::Recognizer::get_postprocessor)
            .def("predict",
                 [](ocr::Recognizer& self, pybind11::array& image) {
                     auto mat = pyarray_to_cv_mat(image);
                     OCRResult ocr_result;
                     self.predict(mat, &ocr_result);
                     return ocr_result;
                 }, pybind11::arg("image"))
            .def("batch_predict", [](ocr::Recognizer& self,
                                     std::vector<pybind11::array>& images) {
                std::vector<cv::Mat> _images;
                for (auto& image : images) {
                    _images.push_back(pyarray_to_cv_mat(image));
                }
                OCRResult ocr_result;
                self.batch_predict(_images, &ocr_result);
                return ocr_result;
            }, pybind11::arg("images"));
    }
} // namespace modeldeploy
