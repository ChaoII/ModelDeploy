//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    void bind_visualize(pybind11::module& m) {
        m.def("vis_cls",
              [](const pybind11::array& im_data,
                 const ClassifyResult& result,
                 const int top_k,
                 const float threshold,
                 const std::string& font_path = "",
                 const int font_size = 14,
                 const double alpha = 0.5,
                 const bool save_result = false) {
                  auto im = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&im);
                  const auto vis_im = vis_cls(image_data, result, top_k, threshold, font_path,
                                              font_size, alpha,
                                              save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("top_k"),
              pybind11::arg("threshold"), pybind11::arg("font_path") = ""
              , pybind11::arg("font_size") = 14, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);


        m.def("vis_det",
              [](pybind11::array& im_data, const std::vector<DetectionResult>& result,
                 const double threshold = 0.5, const std::string& font_path = "", const int font_size = 14,
                 const double alpha = 0.5, const bool save_result = false) {
                  auto im = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&im);
                  const ImageData vis_im = vis_det(image_data, result, threshold, font_path, font_size, alpha,
                                                   save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("threshold") = 0.5,
              pybind11::arg("font_path") = "", pybind11::arg("font_size") = 14, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);

        m.def("vis_iseg", [](pybind11::array& im_data, const std::vector<InstanceSegResult>& result,
                             const double threshold = 0.5,
                             const std::string& font_path = "", const int font_size = 14,
                             const double alpha = 0.5, const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_iseg(image_data, result, threshold, font_path, font_size, alpha,
                                                    save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("threshold") = 0.5,
              pybind11::arg("font_path") = "", pybind11::arg("font_size") = 14, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);
        m.def("vis_obb", [](pybind11::array& im_data, const std::vector<ObbResult>& result,
                            const double threshold = 0.5,
                            const std::string& font_path = "", const int font_size = 14,
                            const double alpha = 0.5, const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_obb(image_data, result, threshold, font_path, font_size, alpha,
                                                   save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("threshold") = 0.5,
              pybind11::arg("font_path") = "", pybind11::arg("font_size") = 14, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);

        m.def("vis_ocr", [](pybind11::array& im_data, const OCRResult& result,
                            const std::string& font_path = "", const int font_size = 14,
                            const double alpha = 0.5, const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_ocr(image_data, result, font_path, font_size, alpha, save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("font_path") = "",
              pybind11::arg("font_size") = 14, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);


        m.def("vis_det_landmarks", [](pybind11::array& im_data,
                                      const std::vector<DetectionLandmarkResult>& result,
                                      const std::string& font_path = "", const int font_size = 14,
                                      const int landmark_radius = 4, const double alpha = 0.5,
                                      const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_det_landmarks(image_data, result, font_path, font_size, landmark_radius,
                                                             alpha, save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("font_path") = "",
              pybind11::arg("font_size") = 14, pybind11::arg("landmark_radius") = 4, pybind11::arg("alpha") = 0.15,
              pybind11::arg("save_result") = false);

        m.def("vis_lpr", [](pybind11::array& im_data,
                            const std::vector<LprResult>& result,
                            const std::string& font_path = "", const int font_size = 14,
                            const int landmark_radius = 4, const double alpha = 0.5,
                            const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_lpr(image_data, result, font_path, font_size, landmark_radius, alpha,
                                                   save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("font_path") = "",
              pybind11::arg("font_size") = 14, pybind11::arg("landmark_radius") = 4,
              pybind11::arg("alpha") = 0.15, pybind11::arg("save_result") = false);

        m.def("vis_pose", [](pybind11::array& im_data,
                             const std::vector<PoseResult>& result,
                             const std::string& font_path = "", const int font_size = 14,
                             const int landmark_radius = 4, const double alpha = 0.5,
                             const bool save_result = false) {
                  auto cv_image = pyarray_to_cv_mat(im_data);
                  auto image_data = ImageData::from_mat(&cv_image);
                  const ImageData vis_im = vis_pose(image_data, result, font_path, font_size, landmark_radius, alpha,
                                                    save_result);
                  cv::Mat mat;
                  vis_im.to_mat(&mat);
                  return cv_mat_to_pyarray(mat);
              }, pybind11::arg("image"), pybind11::arg("result"), pybind11::arg("font_path") = "",
              pybind11::arg("font_size") = 14, pybind11::arg("landmark_radius") = 4,
              pybind11::arg("alpha") = 0.15, pybind11::arg("save_result") = false);
    }
} // namespace modeldeploy
