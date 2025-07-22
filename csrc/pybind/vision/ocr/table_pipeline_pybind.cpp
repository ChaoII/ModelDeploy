//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/ocr/ppstructurev2_table.h"

namespace modeldeploy::vision {
    void bind_table_pipeline(const pybind11::module& m) {
        // PPStructureV2Table(const std::string& det_model_file,
        //            const std::string& rec_model_file,
        //            const std::string& table_model_file,
        //            const std::string& rec_label_file,
        //            const std::string& table_char_dict_path,
        //            int max_side_len = 960,
        //            double det_db_thresh = 0.3,
        //            double det_db_box_thresh = 0.6,
        //            double det_db_unclip_ratio = 1.5,
        //            const std::string& det_db_score_mode = "slow",
        //            bool use_dilation = false,
        //            int rec_batch_size = 8, const RuntimeOption& option = RuntimeOption());


        pybind11::class_<ocr::PPStructureV2Table, BaseModel>(m, "PPStructureV2Table")
            .def(pybind11::init<std::string, std::string, std::string, std::string, std::string,
                                int, double, double, double, std::string, bool, int, RuntimeOption>(),
                 pybind11::arg("det_model_file"),
                 pybind11::arg("rec_model_file"),
                 pybind11::arg("table_model_file"),
                 pybind11::arg("rec_label_file"),
                 pybind11::arg("table_char_dict_path"),
                 pybind11::arg("max_side_len") = 960,
                 pybind11::arg("det_db_thresh") = 0.3,
                 pybind11::arg("det_db_box_thresh") = 0.6,
                 pybind11::arg("det_db_unclip_ratio") = 1.5,
                 pybind11::arg("det_db_score_mode") = "slow",
                 pybind11::arg("use_dilation") = false,
                 pybind11::arg("rec_batch_size") = 8,
                 pybind11::arg("option"))
            .def("predict",
                 [](ocr::PPStructureV2Table& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     OCRResult result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))

            .def("batch_predict",
                 [](ocr::PPStructureV2Table& self, std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<OCRResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property("rec_batch_size",
                          &ocr::PPStructureV2Table::get_rec_batch_size,
                          &ocr::PPStructureV2Table::set_rec_batch_size);
    }
} // namespace modeldeploy
