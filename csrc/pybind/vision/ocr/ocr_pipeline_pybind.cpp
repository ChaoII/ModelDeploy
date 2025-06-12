//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/ocr/ppocr.h"

namespace modeldeploy::vision {
    // PaddleOCR(const std::string& det_model_path,
    //       const std::string& cls_model_path,
    //       const std::string& rec_model_path,
    //       const std::string& dict_path,
    //       int max_side_len = 960,
    //       double det_db_thresh = 0.3,
    //       double det_db_box_thresh = 0.6,
    //       double det_db_unclip_ratio = 1.5,
    //       const std::string& det_db_score_mode = "slow",
    //       bool use_dilation = false,
    //       int rec_batch_size = 6, const RuntimeOption& option = RuntimeOption());

    void bind_ocr_pipeline(const pybind11::module& m) {
        pybind11::class_<ocr::PaddleOCR, BaseModel>(m, "PaddleOCR")
            .def(pybind11::init<std::string, std::string, std::string, std::string, int, double, double, double,
                                std::string, bool, int, RuntimeOption>(),
                 pybind11::arg("det_model_path"),
                 pybind11::arg("cls_model_path"),
                 pybind11::arg("rec_model_path"),
                 pybind11::arg("dict_path"),
                 pybind11::arg("max_side_len") = 960,
                 pybind11::arg("det_db_thresh") = 0.3,
                 pybind11::arg("det_db_box_thresh") = 0.6,
                 pybind11::arg("det_db_unclip_ratio") = 1.5,
                 pybind11::arg("det_db_score_mode") = "slow",
                 pybind11::arg("use_dilation") = false,
                 pybind11::arg("rec_batch_size") = 6,
                 pybind11::arg("option"))
            .def("predict",
                 [](ocr::PaddleOCR& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     OCRResult result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"))

            .def("batch_predict",
                 [](ocr::PaddleOCR& self, std::vector<pybind11::array>& images) {
                     std::vector<cv::Mat> _images;
                     for (auto& image : images) {
                         _images.push_back(pyarray_to_cv_mat(image));
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
                          &ocr::PaddleOCR::set_rec_batch_size);
    }
} // namespace modeldeploy
