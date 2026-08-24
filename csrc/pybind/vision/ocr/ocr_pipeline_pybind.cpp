//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/ocr/ppocr.h"

namespace modeldeploy::vision {
    void bind_ocr_pipeline(const pybind11::module& m) {
        pybind11::class_<ocr::PaddleOCR, BaseModel>(m, "PaddleOCR")
            .def(pybind11::init([](const std::filesystem::path& det_model_path, const std::filesystem::path& cls_model_path, const std::filesystem::path& rec_model_path, const std::filesystem::path& dict_path, const RuntimeOption& option) {
                return std::make_unique<ocr::PaddleOCR>(det_model_path.string(), cls_model_path.string(), rec_model_path.string(), dict_path.string(), option);
            }),
                 pybind11::arg("det_model_path"),
                 pybind11::arg("cls_model_path"),
                 pybind11::arg("rec_model_path"),
                 pybind11::arg("dict_path"),
                 pybind11::arg("option"))
            .def("predict",
                 [](ocr::PaddleOCR& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     OCRResult result;
                     self.predict(ImageData(mat), &result);
                     return result;
                 }, pybind11::arg("image"))

            .def("batch_predict",
                 [](ocr::PaddleOCR& self, std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData(cv_image));
                     }
                     std::vector<OCRResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property("cls_batch_size",
                          &ocr::PaddleOCR::get_cls_batch_size,
                          &ocr::PaddleOCR::set_cls_batch_size)
            .def_property("rec_batch_size",
                          &ocr::PaddleOCR::get_rec_batch_size,
                          &ocr::PaddleOCR::set_rec_batch_size)
            .def("get_detector", &ocr::PaddleOCR::get_detector)
            .def("get_classifier", &ocr::PaddleOCR::get_classifier)
            .def("get_recognizer", &ocr::PaddleOCR::get_recognizer)
            .def("clone", [](const ocr::PaddleOCR& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
