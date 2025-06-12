//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/ocr/ppocr.h"

namespace modeldeploy::vision {
    void bind_ocr_pipeline(const pybind11::module& m) {
        pybind11::class_<ocr::PaddleOCR, BaseModel>(m, "PaddleOCR")
            .def(pybind11::init<std::string, std::string, std::string, std::string, int, double, double, double,
                                std::string, bool, int, RuntimeOption>())
            .def("predict",
                 [](ocr::PaddleOCR& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     OCRResult result;
                     self.predict(mat, &result);
                     return result;
                 })

            .def("batch_predict",
                 [](ocr::PaddleOCR& self, std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<OCRResult> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property("cls_batch_size",
                          &ocr::PaddleOCR::get_cls_batch_size,
                          &ocr::PaddleOCR::set_cls_batch_size)
            .def_property("rec_batch_size",
                          &ocr::PaddleOCR::get_rec_batch_size,
                          &ocr::PaddleOCR::set_rec_batch_size);
    }
} // namespace modeldeploy
