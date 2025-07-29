//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/ocr/ppocr.h"


namespace modeldeploy::vision {
    void bind_ocr_cls(const pybind11::module& m) {
        // Classifier
        pybind11::class_<ocr::ClassifierPreprocessor>(m, "ClassifierPreprocessor")
            .def(pybind11::init<>())
            .def_property("cls_image_shape",
                          &ocr::ClassifierPreprocessor::get_cls_image_shape,
                          &ocr::ClassifierPreprocessor::set_cls_image_shape)
            .def("set_normalize",
                 [](ocr::ClassifierPreprocessor& self,
                    const std::vector<float>& mean, const std::vector<float>& std,
                    const bool is_scale) {
                     self.set_normalize(mean, std, is_scale);
                 }, pybind11::arg("mean"), pybind11::arg("std"), pybind11::arg("is_scale"))
            .def("run",
                 [](ocr::ClassifierPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.apply(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in ClassifierPreprocessor.");
                     }
                     return outputs;
                 }, pybind11::arg("im_list"));

        pybind11::class_<ocr::ClassifierPostprocessor>(
                m, "ClassifierPostprocessor")
            .def(pybind11::init<>())
            .def_property("cls_thresh",
                          &ocr::ClassifierPostprocessor::get_cls_thresh,
                          &ocr::ClassifierPostprocessor::set_cls_thresh)
            .def("run",
                 [](ocr::ClassifierPostprocessor& self,
                    std::vector<Tensor>& inputs) {
                     std::vector<int> cls_labels;
                     std::vector<float> cls_scores;
                     if (!self.run(inputs, &cls_labels, &cls_scores)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "ClassifierPostprocessor.");
                     }
                     return std::make_pair(cls_labels, cls_scores);
                 }, pybind11::arg("inputs"))
            .def("run", [](ocr::ClassifierPostprocessor& self,
                           std::vector<pybind11::array>& input_array) {
                std::vector<Tensor> inputs;
                pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                std::vector<int> cls_labels;
                std::vector<float> cls_scores;
                if (!self.run(inputs, &cls_labels, &cls_scores)) {
                    throw std::runtime_error(
                        "Failed to preprocess the input data in "
                        "ClassifierPostprocessor.");
                }
                return std::make_pair(cls_labels, cls_scores);
            }, pybind11::arg("inputs"));

        pybind11::class_<ocr::Classifier, BaseModel>(m, "Classifier")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::Classifier::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::Classifier::get_postprocessor)
            .def("predict",
                 [](ocr::Classifier& self, pybind11::array& image) {
                     auto mat = pyarray_to_cv_mat(image);
                     OCRResult ocr_result;
                     self.predict(ImageData::from_mat(&mat), &ocr_result);
                     return ocr_result;
                 }, pybind11::arg("image"))
            .def("batch_predict", [](ocr::Classifier& self,
                                     std::vector<pybind11::array>& images) {
                std::vector<ImageData> _images;
                for (auto& image : images) {
                    auto cv_image = pyarray_to_cv_mat(image);
                    _images.push_back(ImageData::from_mat(&cv_image));
                }
                OCRResult ocr_result;
                self.batch_predict(_images, &ocr_result);
                return ocr_result;
            }, pybind11::arg("images"));
    }
} // namespace modeldeploy
