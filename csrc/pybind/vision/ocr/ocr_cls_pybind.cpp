//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/ocr/ppocr.h"


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
                 })
            .def("run",
                 [](ocr::ClassifierPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     for (auto& image : im_list) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.apply(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "ClassifierPreprocessor.");
                     }
                     pybind11::array array;
                     tensor_list_to_pyarray(outputs, array);
                     return array;
                 });

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
                 })
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
            });

        pybind11::class_<ocr::Classifier, BaseModel>(m, "Classifier")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::Classifier::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::Classifier::get_postprocessor)
            .def("predict",
                 [](ocr::Classifier& self, pybind11::array& data) {
                     auto mat = pyarray_to_cv_mat(data);
                     OCRResult ocr_result;
                     self.predict(mat, &ocr_result);
                     return ocr_result;
                 })
            .def("batch_predict", [](ocr::Classifier& self,
                                     std::vector<pybind11::array>& data) {
                std::vector<cv::Mat> images;
                for (auto& image : data) {
                    images.push_back(pyarray_to_cv_mat(image));
                }
                OCRResult ocr_result;
                self.batch_predict(images, &ocr_result);
                return ocr_result;
            });
    }
} // namespace modeldeploy
