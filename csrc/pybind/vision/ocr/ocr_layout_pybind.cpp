//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"

#include "csrc/vision/ocr/structurev2_layout.h"
#include "csrc/vision/ocr/structurev2_ser_vi_layoutxlm.h"


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
                 })
            .def("run",
                 [](ocr::StructureV2LayoutPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     for (size_t i = 0; i < im_list.size(); ++i) {
                         images.push_back(pyarray_to_cv_mat(im_list[i]));
                     }
                     std::vector<Tensor> outputs;
                     if (!self.run(&images, &outputs)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "StructureV2LayoutPreprocessor.");
                     }
                     auto batch_layout_img_info = self.get_batch_layout_image_info();
                     return std::make_pair(outputs, *batch_layout_img_info);
                 });


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
                 });

        pybind11::class_<ocr::StructureV2Layout, BaseModel>(
                m, "StructureV2Layout")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::StructureV2Layout::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::StructureV2Layout::get_postprocessor)
            .def("predict",
                 [](ocr::StructureV2Layout& self, pybind11::array& data) {
                     auto mat = pyarray_to_cv_mat(data);
                     std::vector<DetectionResult> result;
                     self.predict(mat, &result);
                     return result;
                 })
            .def("batch_predict", [](ocr::StructureV2Layout& self,
                                     std::vector<pybind11::array>& data) {
                std::vector<cv::Mat> images;
                for (size_t i = 0; i < data.size(); ++i) {
                    images.push_back(pyarray_to_cv_mat(data[i]));
                }
                std::vector<std::vector<DetectionResult>> results;
                self.batch_predict(images, &results);
                return results;
            });
        pybind11::class_<ocr::StructureV2SERViLayoutXLMModel, BaseModel>(m, "StructureV2SERViLayoutXLMModel")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](ocr::StructureV2SERViLayoutXLMModel& self,
                    pybind11::array& data) {
                     throw std::runtime_error(
                         "StructureV2SERViLayoutXLMModel do not support predict.");
                 })
            .def(
                "batch_predict",
                [](ocr::StructureV2SERViLayoutXLMModel& self,
                   std::vector<pybind11::array>& data) {
                    throw std::runtime_error(
                        "StructureV2SERViLayoutXLMModel do not support batch_predict.");
                })
            .def("infer",
                 [](ocr::StructureV2SERViLayoutXLMModel& self,
                    std::map<std::string, pybind11::array>& data) {
                     std::vector<Tensor> inputs(data.size());
                     int index = 0;
                     for (auto iter = data.begin(); iter != data.end(); ++iter) {
                         std::vector<int64_t> data_shape;
                         data_shape.insert(data_shape.begin(), iter->second.shape(),
                                           iter->second.shape() + iter->second.ndim());
                         auto dtype = numpy_data_type_to_md_data_type(iter->second.dtype());

                         inputs[index].resize(data_shape, dtype);
                         memcpy(inputs[index].data(), iter->second.mutable_data(),
                                iter->second.nbytes());
                         inputs[index].set_name(iter->first);
                         index += 1;
                     }

                     std::vector<Tensor> outputs(self.num_outputs());
                     self.infer(inputs, &outputs);

                     std::vector<pybind11::array> results;
                     results.reserve(outputs.size());
                     for (size_t i = 0; i < outputs.size(); ++i) {
                         auto numpy_dtype = md_data_type_to_numpy_data_type(outputs[i].dtype());
                         results.emplace_back(
                             pybind11::array(numpy_dtype, outputs[i].shape()));
                         memcpy(results[i].mutable_data(), outputs[i].data(),
                                outputs[i].size() * Tensor::get_element_size(outputs[i].dtype()));
                     }
                     return results;
                 })
            .def("get_input_info",
                 [](ocr::StructureV2SERViLayoutXLMModel& self, const int& index) {
                     return self.get_input_info(index);
                 });
    }
} // namespace modeldeploy
