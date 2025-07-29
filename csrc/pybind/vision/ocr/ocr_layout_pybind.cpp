//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"

#include "vision/ocr/structurev2_layout.h"
#include "vision/ocr/structurev2_ser_vi_layoutxlm.h"


namespace modeldeploy::vision {
    void bind_ocr_layout(const pybind11::module& m) {
        // Layout
        pybind11::class_<ocr::StructureV2LayoutPreprocessor>(m, "StructureV2LayoutPreprocessor")
            .def(pybind11::init<>())
            .def_property(
                "static_shape_infer",
                &ocr::StructureV2LayoutPreprocessor::get_static_shape_infer,
                &ocr::StructureV2LayoutPreprocessor::set_static_shape_infer)
            .def_property(
                "layout_image_shape",
                &ocr::StructureV2LayoutPreprocessor::get_layout_image_shape,
                &ocr::StructureV2LayoutPreprocessor::set_layout_image_shape)
            .def("set_normalize",
                 [](ocr::StructureV2LayoutPreprocessor& self,
                    const std::vector<float>& mean, const std::vector<float>& std,
                    const bool is_scale) {
                     self.set_normalize(mean, std, is_scale);
                 }, pybind11::arg("mean"), pybind11::arg("std"), pybind11::arg("is_scale"))
            .def("run",
                 [](ocr::StructureV2LayoutPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "StructureV2LayoutPreprocessor.");
                     }
                     auto batch_layout_img_info = self.get_batch_layout_image_info();
                     return std::make_pair(outputs, *batch_layout_img_info);
                 }, pybind11::arg("im_list"));


        pybind11::class_<ocr::StructureV2LayoutPostprocessor>(
                m, "StructureV2LayoutPostprocessor")
            .def(pybind11::init<>())
            .def_property(
                "score_threshold",
                &ocr::StructureV2LayoutPostprocessor::get_score_threshold,
                &ocr::StructureV2LayoutPostprocessor::set_score_threshold)
            .def_property(
                "nms_threshold",
                &ocr::StructureV2LayoutPostprocessor::get_nms_threshold,
                &ocr::StructureV2LayoutPostprocessor::set_nms_threshold)
            .def_property("num_class",
                          &ocr::StructureV2LayoutPostprocessor::get_num_class,
                          &ocr::StructureV2LayoutPostprocessor::set_num_class)
            .def_property("fpn_stride",
                          &ocr::StructureV2LayoutPostprocessor::get_fpn_stride,
                          &ocr::StructureV2LayoutPostprocessor::set_fpn_stride)
            .def_property("reg_max",
                          &ocr::StructureV2LayoutPostprocessor::get_reg_max,
                          &ocr::StructureV2LayoutPostprocessor::set_reg_max)
            .def("run",
                 [](ocr::StructureV2LayoutPostprocessor& self,
                    std::vector<Tensor>& inputs,
                    const std::vector<std::array<int, 4>>& batch_layout_img_info) {
                     std::vector<std::vector<DetectionResult>> results;

                     if (!self.run(inputs, &results, batch_layout_img_info)) {
                         throw std::runtime_error(
                             "Failed to postprocess the input data in "
                             "StructureV2LayoutPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("batch_layout_img_info"));

        pybind11::class_<ocr::StructureV2Layout, BaseModel>(
                m, "StructureV2Layout")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::StructureV2Layout::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::StructureV2Layout::get_postprocessor)
            .def("predict",
                 [](ocr::StructureV2Layout& self, pybind11::array& image) {
                     auto mat = pyarray_to_cv_mat(image);
                     std::vector<DetectionResult> result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict", [](ocr::StructureV2Layout& self,
                                     std::vector<pybind11::array>& images) {
                std::vector<ImageData> _images;
                for (auto& image : images) {
                    auto cv_image = pyarray_to_cv_mat(image);
                    _images.push_back(ImageData::from_mat(&cv_image));
                }
                std::vector<std::vector<DetectionResult>> results;
                self.batch_predict(_images, &results);
                return results;
            }, pybind11::arg("images"));
        pybind11::class_<ocr::StructureV2SERViLayoutXLMModel, BaseModel>(m, "StructureV2SERViLayoutXLMModel")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](ocr::StructureV2SERViLayoutXLMModel& self,
                    pybind11::array& image) {
                     throw std::runtime_error(
                         "StructureV2SERViLayoutXLMModel do not support predict.");
                 }, pybind11::arg("image"))
            .def(
                "batch_predict",
                [](ocr::StructureV2SERViLayoutXLMModel& self,
                   std::vector<pybind11::array>& images) {
                    throw std::runtime_error(
                        "StructureV2SERViLayoutXLMModel do not support batch_predict.");
                }, pybind11::arg("images"))
            .def("infer",
                 [](ocr::StructureV2SERViLayoutXLMModel& self,
                    std::map<std::string, pybind11::array>& inputs) {
                     std::vector<Tensor> _inputs(inputs.size());
                     int index = 0;
                     for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
                         std::vector<int64_t> data_shape;
                         data_shape.insert(data_shape.begin(), iter->second.shape(),
                                           iter->second.shape() + iter->second.ndim());
                         auto dtype = numpy_data_type_to_md_data_type(iter->second.dtype());

                         _inputs[index].resize(data_shape, dtype);
                         memcpy(_inputs[index].data(), iter->second.mutable_data(),
                                iter->second.nbytes());
                         _inputs[index].set_name(iter->first);
                         index += 1;
                     }
                     std::vector<Tensor> outputs(self.num_outputs());
                     self.infer(_inputs, &outputs);
                     std::vector<pybind11::array> results;
                     results.reserve(outputs.size());
                     for (size_t i = 0; i < outputs.size(); ++i) {
                         auto numpy_dtype = md_data_type_to_numpy_data_type(outputs[i].dtype());
                         results.emplace_back(numpy_dtype, outputs[i].shape());
                         memcpy(results[i].mutable_data(), outputs[i].data(),
                                outputs[i].size() * Tensor::get_element_size(outputs[i].dtype()));
                     }
                     return results;
                 }, pybind11::arg("inputs"))
            .def("get_input_info",
                 [](const ocr::StructureV2SERViLayoutXLMModel& self, const int& index) {
                     return self.get_input_info(index);
                 }, pybind11::arg("index"));
    }
} // namespace modeldeploy
