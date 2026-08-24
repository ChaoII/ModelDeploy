//
// insightface buffalo_l pybind 绑定。
//

#include "pybind/utils/utils.h"
#include "vision/face/insightface/face_analysis.h"

namespace modeldeploy::vision {
    void bind_insightface(const pybind11::module& m) {
        // 结果类型
        pybind11::class_<face::InsightFaceBox>(m, "InsightFaceBox")
            .def(pybind11::init<>())
            .def_readwrite("bbox", &face::InsightFaceBox::bbox)
            .def_readwrite("kps", &face::InsightFaceBox::kps)
            .def_readwrite("score", &face::InsightFaceBox::score);

        pybind11::class_<face::InsightFaceResult>(m, "InsightFaceResult")
            .def(pybind11::init<>())
            .def_readwrite("bbox", &face::InsightFaceResult::bbox)
            .def_readwrite("det_score", &face::InsightFaceResult::det_score)
            .def_readwrite("kps", &face::InsightFaceResult::kps)
            .def_readwrite("landmark_2d_106", &face::InsightFaceResult::landmark_2d_106)
            .def_readwrite("landmark_3d_68", &face::InsightFaceResult::landmark_3d_68)
            .def_readwrite("pose", &face::InsightFaceResult::pose)
            .def_readwrite("embedding", &face::InsightFaceResult::embedding)
            .def_readwrite("gender", &face::InsightFaceResult::gender)
            .def_readwrite("age", &face::InsightFaceResult::age);

        // 检测模型
        pybind11::class_<face::InsightFaceDetPreprocessor>(m, "InsightFaceDetPreprocessor")
            .def(pybind11::init<>())
            .def_property("size", &face::InsightFaceDetPreprocessor::get_size,
                          &face::InsightFaceDetPreprocessor::set_size);

        pybind11::class_<face::InsightFaceDet, BaseModel>(m, "InsightFaceDet")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<face::InsightFaceDet>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](face::InsightFaceDet& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<face::InsightFaceBox> result;
                     self.predict(ImageData(mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def_property_readonly("preprocessor", &face::InsightFaceDet::get_preprocessor)
            .def_property_readonly("postprocessor", &face::InsightFaceDet::get_postprocessor);

        // 关键点模型
        pybind11::class_<face::InsightFaceLandmark, BaseModel>(m, "InsightFaceLandmark")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<face::InsightFaceLandmark>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict_2d106",
                 [](face::InsightFaceLandmark& self, const pybind11::array& image,
                    const std::array<float, 4>& bbox) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<std::array<float, 2>> landmarks;
                     self.predict_2d106(ImageData(mat), bbox, &landmarks);
                     return landmarks;
                 }, pybind11::arg("image"), pybind11::arg("bbox"))
            .def("predict_3d68",
                 [](face::InsightFaceLandmark& self, const pybind11::array& image,
                    const std::array<float, 4>& bbox) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<std::array<float, 3>> landmarks;
                     std::array<float, 3> pose{0, 0, 0};
                     self.predict_3d68(ImageData(mat), bbox, &landmarks, &pose);
                     return pybind11::make_tuple(landmarks, pose);
                 }, pybind11::arg("image"), pybind11::arg("bbox"));

        // 识别模型
        pybind11::class_<face::InsightFaceRecognition, BaseModel>(m, "InsightFaceRecognition")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<face::InsightFaceRecognition>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](face::InsightFaceRecognition& self, const pybind11::array& image,
                    const std::vector<std::array<float, 2>>& kps) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<float> embedding;
                     self.predict(ImageData(mat), kps, &embedding);
                     return embedding;
                 }, pybind11::arg("image"), pybind11::arg("kps"));

        // 性别年龄模型
        pybind11::class_<face::InsightFaceGenderAge, BaseModel>(m, "InsightFaceGenderAge")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<face::InsightFaceGenderAge>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict_gender_age",
                 [](face::InsightFaceGenderAge& self, const pybind11::array& image,
                    const std::array<float, 4>& bbox) {
                     const auto mat = pyarray_to_cv_mat(image);
                     face::GenderAgeResult result;
                     self.predict_gender_age(ImageData(mat), bbox, &result);
                     return pybind11::make_tuple(result.gender, result.age);
                 }, pybind11::arg("image"), pybind11::arg("bbox"));

        // 综合 pipeline
        pybind11::class_<face::InsightFaceAnalysis>(m, "InsightFaceAnalysis")
            .def(pybind11::init([](const std::filesystem::path& det_model, const std::filesystem::path& rec_model, const std::filesystem::path& lmk2d_model, const std::filesystem::path& lmk3d_model, const RuntimeOption& option, const std::filesystem::path& genderage_model) {
                return std::make_unique<face::InsightFaceAnalysis>(det_model.string(), rec_model.string(), lmk2d_model.string(), lmk3d_model.string(), option, genderage_model.string());
            }), pybind11::arg("det_model"), pybind11::arg("rec_model"),
                 pybind11::arg("lmk2d_model"), pybind11::arg("lmk3d_model"),
                 pybind11::arg("option") = RuntimeOption(),
                 pybind11::arg("genderage_model") = std::filesystem::path())
            .def_static("create_from_dir",
                        [](const std::filesystem::path& model_dir, const RuntimeOption& option) {
                            return face::InsightFaceAnalysis::create_from_dir(model_dir.string(), option);
                        }, pybind11::arg("model_dir"), pybind11::arg("option") = RuntimeOption())
            .def("analyze",
                 [](face::InsightFaceAnalysis& self, const pybind11::array& image,
                    bool with_2d106, bool with_3d68, bool with_recognition, bool with_genderage) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<face::InsightFaceResult> results;
                     self.analyze(ImageData(mat), &results,
                                  with_2d106, with_3d68, with_recognition, with_genderage);
                     return results;
                 }, pybind11::arg("image"),
                 pybind11::arg("with_2d106") = true,
                 pybind11::arg("with_3d68") = true,
                 pybind11::arg("with_recognition") = true,
                 pybind11::arg("with_genderage") = true)
            .def("detect",
                 [](face::InsightFaceAnalysis& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<face::InsightFaceBox> boxes;
                     self.detect(ImageData(mat), &boxes);
                     return boxes;
                 }, pybind11::arg("image"))
            .def("set_det_thresh", &face::InsightFaceAnalysis::set_det_thresh)
            .def_property_readonly("genderage", &face::InsightFaceAnalysis::genderage);
    }
} // namespace modeldeploy
