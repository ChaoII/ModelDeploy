//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/ocr/ppocr.h"


namespace modeldeploy::vision {
    void bind_ocr_db(const pybind11::module& m) {
        // DBDetector
        pybind11::class_<ocr::DBDetectorPreprocessor>(m, "DBDetectorPreprocessor")
            .def(pybind11::init<>())
            .def_property("static_shape_infer",
                          &ocr::DBDetectorPreprocessor::get_static_shape_infer,
                          &ocr::DBDetectorPreprocessor::set_static_shape_infer)
            .def_property("max_side_len",
                          &ocr::DBDetectorPreprocessor::get_max_side_len,
                          &ocr::DBDetectorPreprocessor::set_max_side_len)
            .def("set_normalize",
                 [](ocr::DBDetectorPreprocessor& self,
                    const std::vector<float>& mean, const std::vector<float>& std,
                    bool is_scale) {
                     self.set_normalize(mean, std, is_scale);
                 })
            .def("run",
                 [](ocr::DBDetectorPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<cv::Mat> images;
                     for (size_t i = 0; i < im_list.size(); ++i) {
                         images.push_back(pyarray_to_cv_mat(im_list[i]));
                     }
                     std::vector<Tensor> outputs;
                     self.apply(&images, &outputs);
                     auto batch_det_img_info = self.get_batch_img_info();
                     return std::make_pair(outputs, *batch_det_img_info);
                 });


        pybind11::class_<ocr::DBDetectorPostprocessor>(
                m, "DBDetectorPostprocessor")
            .def(pybind11::init<>())
            .def_property("det_db_thresh",
                          &ocr::DBDetectorPostprocessor::get_det_db_thresh,
                          &ocr::DBDetectorPostprocessor::set_det_db_thresh)
            .def_property("det_db_box_thresh",
                          &ocr::DBDetectorPostprocessor::get_det_db_box_thresh,
                          &ocr::DBDetectorPostprocessor::set_det_db_box_thresh)
            .def_property("det_db_unclip_ratio",
                          &ocr::DBDetectorPostprocessor::get_det_db_unclip_ratio,
                          &ocr::DBDetectorPostprocessor::set_det_db_unclip_ratio)
            .def_property("det_db_score_mode",
                          &ocr::DBDetectorPostprocessor::get_det_db_score_mode,
                          &ocr::DBDetectorPostprocessor::set_det_db_score_mode)
            .def_property("use_dilation",
                          &ocr::DBDetectorPostprocessor::get_use_dilation,
                          &ocr::DBDetectorPostprocessor::set_use_dilation)

            .def("run",
                 [](ocr::DBDetectorPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<std::array<int, 4>>& batch_det_img_info) {
                     std::vector<std::vector<std::array<int, 8>>> results;
                     if (!self.apply(inputs, &results, batch_det_img_info)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "DBDetectorPostprocessor.");
                     }
                     return results;
                 })
            .def("run",
                 [](ocr::DBDetectorPostprocessor& self,
                    std::vector<pybind11::array>& input_array,
                    const std::vector<std::array<int, 4>>& batch_det_img_info) {
                     std::vector<std::vector<std::array<int, 8>>> results;
                     std::vector<Tensor> inputs;
                     pyarray_to_tensor_list(input_array, &inputs, /*share_buffer=*/true);
                     if (!self.apply(inputs, &results, batch_det_img_info)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in "
                             "DBDetectorPostprocessor.");
                     }
                     return results;
                 });

        pybind11::class_<ocr::DBDetector, BaseModel>(m, "DBDetector")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def(pybind11::init<>())
            .def_property_readonly("preprocessor",
                                   &ocr::DBDetector::get_preprocessor)
            .def_property_readonly("postprocessor",
                                   &ocr::DBDetector::get_postprocessor)
            .def("predict",
                 [](ocr::DBDetector& self, pybind11::array& data) {
                     auto mat = pyarray_to_cv_mat(data);
                     OCRResult ocr_result;
                     self.predict(mat, &ocr_result);
                     return ocr_result;
                 })
            .def("batch_predict", [](ocr::DBDetector& self,
                                     std::vector<pybind11::array>& data) {
                std::vector<cv::Mat> images;
                for (size_t i = 0; i < data.size(); ++i) {
                    images.push_back(pyarray_to_cv_mat(data[i]));
                }
                std::vector<OCRResult> ocr_results;
                self.batch_predict(images, &ocr_results);
                return ocr_results;
            });
    }
} // namespace modeldeploy
