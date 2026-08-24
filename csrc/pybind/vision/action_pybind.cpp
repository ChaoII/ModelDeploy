#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vision/action/keypoint_seq.h"
#include "vision/action/tsn.h"
#include "vision/action/st_gcn.h"
#include "vision/common/image_data.h"
#include "pybind/utils/utils.h"

namespace modeldeploy::vision {
    void bind_action(pybind11::module& m) {
        // modeldeploy.vision.action：视频动作识别（TSN / ST-GCN 骨骼）模型绑定。
        auto action_m = m.def_submodule(
            "action",
            "Video action recognition models of Modeldeploy: TSN (multi-frame RGB) and "
            "StGcn (skeleton graph) action classification.");

        pybind11::class_<action::KeyPointSeq>(action_m, "KeyPointSeq",
                                              "一段视频片段的骨骼序列：frames[t][v] 为第 t 帧第 v 个关节坐标 (Point3f)。")
            .def(pybind11::init<>(), "构造空的骨骼序列。")
            .def_readwrite("frames", &action::KeyPointSeq::frames,
                           "骨骼序列，类型为 List[List[Point3f]]");

        pybind11::class_<action::TSN>(action_m, "TSN",
                                      "TSN 动作识别：多帧 RGB -> 时序聚合 -> 类别 scores。")
            .def(pybind11::init([](const std::filesystem::path& model_file, pybind11::object option_obj) {
                RuntimeOption option = pybind11::none().equal(option_obj) ? RuntimeOption() : option_obj.cast<RuntimeOption>();
                return std::make_unique<action::TSN>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = pybind11::none(),
                 "构造 TSN 模型；无权重时 is_initialized()==False。")
            .def("predict",
                 [](action::TSN& self, const std::vector<ImageData>& frames) {
                     std::vector<float> scores;
                     if (!self.predict(frames, &scores))
                         throw std::runtime_error("TSN predict failed");
                     return scores;
                 },
                 pybind11::arg("frames"),
                 "对多帧 RGB 图像序列推理，返回类别 scores (List[float])。")
            .def("is_initialized", &action::TSN::is_initialized, "模型是否已成功初始化。");

        pybind11::class_<action::StGcn>(action_m, "StGcn",
                                        "ST-GCN 骨骼动作识别：骨骼序列 -> 图卷积 -> 类别 scores。")
            .def(pybind11::init([](const std::filesystem::path& model_file, pybind11::object option_obj) {
                RuntimeOption option = pybind11::none().equal(option_obj) ? RuntimeOption() : option_obj.cast<RuntimeOption>();
                return std::make_unique<action::StGcn>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option") = pybind11::none(),
                 "构造 StGcn 模型；无权重时 is_initialized()==False。")
            .def("predict",
                 [](action::StGcn& self, const action::KeyPointSeq& seq) {
                     std::vector<float> scores;
                     if (!self.predict(seq, &scores))
                         throw std::runtime_error("StGcn predict failed");
                     return scores;
                 },
                 pybind11::arg("seq"),
                 "对骨骼序列推理，返回类别 scores (List[float])。")
            .def("is_initialized", &action::StGcn::is_initialized, "模型是否已成功初始化。");
    }
} // namespace modeldeploy::vision
