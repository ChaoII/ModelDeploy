//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/ocr/ppstructurev2_table.h"

namespace modeldeploy::vision {
    void bind_table_pipeline(const pybind11::module& m) {
        pybind11::class_<ocr::PPStructureV2Table, BaseModel>(m, "PPStructureV2Table")
            .def(pybind11::init<std::string, std::string, std::string, std::string, std::string,
                                int, double, double, double, std::string, bool, int, RuntimeOption>())
            .def("predict",
                 [](ocr::PPStructureV2Table& self, pybind11::array& data) {
                     const auto mat = pyarray_to_cv_mat(data);
                     OCRResult result;
                     self.predict(mat, &result);
                     return result;
                 })

            .def("batch_predict",
                 [](ocr::PPStructureV2Table& self, std::vector<pybind11::array>& data) {
                     std::vector<cv::Mat> images;
                     for (auto& image : data) {
                         images.push_back(pyarray_to_cv_mat(image));
                     }
                     std::vector<OCRResult> results;
                     self.batch_predict(images, &results);
                     return results;
                 })
            .def_property("rec_batch_size",
                          &ocr::PPStructureV2Table::get_rec_batch_size,
                          &ocr::PPStructureV2Table::set_rec_batch_size);
    }
} // namespace modeldeploy
