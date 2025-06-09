//
// Created by aichao on 2025/6/9.
//

#include <csrc/vision/common/display/display.h>

#include "csrc/vision/common/result.h"
#include <pybind11/pybind11.h>

namespace modeldeploy::vision {
    void bind_ultralytics_det(pybind11::module&);

    void bind_vision(pybind11::module& m) {
        pybind11::class_<Mask>(m, "Mask")
            .def(pybind11::init())
            .def_readwrite("buffer", &Mask::buffer)
            .def_readwrite("shape", &Mask::shape)
            .def(pybind11::pickle(
                [](const Mask& mask) {
                    return pybind11::make_tuple(mask.buffer, mask.shape);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 2)
                        throw std::runtime_error(
                            "Mask pickle with invalid state!");
                    Mask mask;
                    mask.buffer = t[0].cast<std::vector<uint8_t>>();
                    mask.shape = t[1].cast<std::vector<int64_t>>();
                    return mask;
                }));

        pybind11::class_<ClassifyResult>(m, "ClassifyResult")
            .def(pybind11::init())
            .def_readwrite("label_ids", &ClassifyResult::label_ids)
            .def_readwrite("scores", &ClassifyResult::scores)
            .def_readwrite("feature", &ClassifyResult::feature)
            .def(pybind11::pickle(
                [](const ClassifyResult& c) {
                    if (c.feature.empty()) {
                        return pybind11::make_tuple(c.label_ids, c.scores);
                    }
                    return pybind11::make_tuple(c.label_ids, c.scores, c.feature);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 2 && t.size() != 3) {
                        throw std::runtime_error(
                            "ClassifyResult pickle with invalid state!");
                    }

                    ClassifyResult c;
                    c.label_ids = t[0].cast<std::vector<int32_t>>();
                    c.scores = t[1].cast<std::vector<float>>();
                    if (t.size() == 3) {
                        c.feature = t[2].cast<std::vector<float>>();
                    }

                    return c;
                }));


        pybind11::class_<DetectionResult>(m, "DetectionResult")
            .def(pybind11::init())
            .def_readwrite("box", &DetectionResult::box)
            .def_readwrite("score", &DetectionResult::score)
            .def_readwrite("label_id", &DetectionResult::label_id)
            .def(pybind11::pickle(
                [](const DetectionResult& d) {
                    return pybind11::make_tuple(d.box, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 3)
                        throw std::runtime_error("DetectionResult pickle with Invalid state!");
                    DetectionResult d;
                    d.box = t[0].cast<cv::Rect2f>();
                    d.score = t[1].cast<float>();
                    d.label_id = t[2].cast<int32_t>();
                    return d;
                }));
        bind_ultralytics_det(m);
    }
}
