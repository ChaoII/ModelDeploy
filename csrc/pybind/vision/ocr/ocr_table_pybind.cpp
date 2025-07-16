//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/ocr/ppstructurev2_table.h"

namespace modeldeploy::vision {
    void bind_ocr_table(const pybind11::module& m) {
        // table
        pybind11::class_<ocr::StructureV2TablePreprocessor>(m, "StructureV2TablePreprocessor")
            .def(pybind11::init<>())
            .def("run", [](ocr::StructureV2TablePreprocessor& self,
                           std::vector<pybind11::array>& im_list) {
                std::vector<cv::Mat> images;
                for (auto& image : im_list) {
                    images.push_back(pyarray_to_cv_mat(image));
                }
                std::vector<Tensor> outputs;
                if (!self.run(&images, &outputs)) {
                    throw std::runtime_error(
                        "Failed to preprocess the input data in "
                        "StructureV2TablePreprocessor.");
                }
                auto batch_det_img_info = self.GetBatchImgInfo();
                return std::make_pair(outputs, *batch_det_img_info);
            }, pybind11::arg("im_list"));

        pybind11::class_<ocr::StructureV2TablePostprocessor>(
                m, "StructureV2TablePostprocessor")
            .def(pybind11::init<std::string>())
            .def("run",
                 [](ocr::StructureV2TablePostprocessor& self,
                    std::vector<Tensor>& inputs,
                    const std::vector<std::array<int, 4>>& batch_det_img_info) {
                     std::vector<std::vector<std::array<int, 8>>> boxes;
                     std::vector<std::vector<std::string>> structure_list;

                     if (!self.run(inputs, &boxes, &structure_list,
                                   batch_det_img_info)) {
                         throw std::runtime_error(
                             "Failed to postprocess the input data in "
                             "StructureV2TablePostprocessor.");
                     }
                     return std::make_pair(boxes, structure_list);
                 }, pybind11::arg("inputs"), pybind11::arg("batch_det_img_info"))
            .def("run",
                 [](ocr::StructureV2TablePostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<std::array<int, 4>>& batch_det_img_info) {
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     std::vector<std::vector<std::array<int, 8>>> boxes;
                     std::vector<std::vector<std::string>> structure_list;

                     if (!self.run(inputs, &boxes, &structure_list,
                                   batch_det_img_info)) {
                         throw std::runtime_error(
                             "Failed to postprocess the input data in "
                             "StructureV2TablePostprocessor.");
                     }
                     return std::make_pair(boxes, structure_list);
                 }, pybind11::arg("inputs"), pybind11::arg("batch_det_img_info"));

        pybind11::class_<ocr::StructureV2Table, BaseModel>(
                m, "StructureV2Table")
            .def(pybind11::init<std::string, std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::StructureV2Table::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::StructureV2Table::get_postprocessor)
            .def("predict",
                 [](ocr::StructureV2Table& self, pybind11::array& image) {
                     auto mat = pyarray_to_cv_mat(image);
                     OCRResult ocr_result;
                     self.predict(mat, &ocr_result);
                     return ocr_result;
                 }, pybind11::arg("image"))
            .def("batch_predict", [](ocr::StructureV2Table& self,
                                     std::vector<pybind11::array>& images) {
                std::vector<cv::Mat> _images;
                for (auto& image : images) {
                    _images.push_back(pyarray_to_cv_mat(image));
                }
                std::vector<OCRResult> ocr_results;
                self.batch_predict(_images, &ocr_results);
                return ocr_results;
            }, pybind11::arg("images"));
    }
} // namespace modeldeploy
