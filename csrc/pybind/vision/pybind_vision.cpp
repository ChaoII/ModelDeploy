//
// Created by aichao on 2025/6/9.
//

#include <csrc/vision/common/display/display.h>

#include "csrc/vision/common/result.h"
#include <pybind11/pybind11.h>

namespace modeldeploy::vision
{
    void bind_ultralytics_det(pybind11::module&);

    void bind_vision(pybind11::module& m) {
        pybind11::class_<Point2f>(m, "Point2f")
            .def(pybind11::init())
            .def_readwrite("x", &Point2f::x)
            .def_readwrite("y", &Point2f::y)
            .def(pybind11::pickle(
                [](const Point2f& point2f) {
                    return pybind11::make_tuple(point2f.x, point2f.y);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 2)
                        throw std::runtime_error(
                            "Mask pickle with invalid state!");
                    Point2f point2f;
                    point2f.x = t[0].cast<float>();
                    point2f.y = t[1].cast<float>();
                    return point2f;
                }))
            .def("__repr__", [](const Point2f& point2f) {
                return point2f.to_string();
            })
            .def("__str__", [](const Point2f& point2f) {
                return point2f.to_string();
            });


        pybind11::class_<Point3f>(m, "Point3f")
            .def(pybind11::init())
            .def_readwrite("x", &Point3f::x)
            .def_readwrite("y", &Point3f::y)
            .def_readwrite("z", &Point3f::z)
            .def(pybind11::pickle(
                [](const Point3f& point3f) {
                    return pybind11::make_tuple(point3f.x, point3f.y, point3f.z);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 3)
                        throw std::runtime_error(
                            "Mask pickle with invalid state!");
                    Point3f point3f;
                    point3f.x = t[0].cast<float>();
                    point3f.y = t[1].cast<float>();
                    point3f.z = t[2].cast<float>();
                    return point3f;
                }))
            .def("__repr__", [](const Point3f& point3f) {
                return point3f.to_string();
            })
            .def("__str__", [](const Point3f& point3f) {
                return point3f.to_string();
            });
        pybind11::class_<Rect2f>(m, "Rect2f")
            .def(pybind11::init())
            .def_readwrite("x", &Rect2f::x)
            .def_readwrite("y", &Rect2f::y)
            .def_readwrite("width", &Rect2f::width)
            .def_readwrite("height", &Rect2f::height)
            .def(pybind11::pickle(
                [](const Rect2f& rect2f) {
                    return pybind11::make_tuple(rect2f.x, rect2f.y, rect2f.width, rect2f.height);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 4)
                        throw std::runtime_error(
                            "Mask pickle with invalid state!");
                    Rect2f rect2f;
                    rect2f.x = t[0].cast<float>();
                    rect2f.y = t[1].cast<float>();
                    rect2f.width = t[2].cast<float>();
                    rect2f.height = t[3].cast<float>();
                    return rect2f;
                }))
            .def("__repr__", [](const Rect2f& rect2f) {
                return rect2f.to_string();
            })
            .def("__str__", [](const Rect2f& rect2f) {
                return rect2f.to_string();
            });

        pybind11::class_<RotatedRect>(m, "RotatedRect")
            .def(pybind11::init())
            .def_readwrite("xc", &RotatedRect::xc)
            .def_readwrite("yc", &RotatedRect::yc)
            .def_readwrite("width", &RotatedRect::width)
            .def_readwrite("height", &RotatedRect::height)
            .def_readwrite("angle", &RotatedRect::angle)
            .def(pybind11::pickle(
                [](const RotatedRect& rotated_rect) {
                    return pybind11::make_tuple(rotated_rect.xc, rotated_rect.yc, rotated_rect.width,
                                                rotated_rect.height, rotated_rect.angle);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 5)
                        throw std::runtime_error(
                            "Mask pickle with invalid state!");
                    RotatedRect rotated_rect;
                    rotated_rect.xc = t[0].cast<float>();
                    rotated_rect.yc = t[1].cast<float>();
                    rotated_rect.width = t[2].cast<float>();
                    rotated_rect.height = t[3].cast<float>();
                    rotated_rect.angle = t[4].cast<float>();
                    return rotated_rect;
                }))
            .def("__repr__", [](const RotatedRect& rotated_rect) {
                return rotated_rect.to_string();
            })
            .def("__str__", [](const RotatedRect& rotated_rect) {
                return rotated_rect.to_string();
            });


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
                    d.box = t[0].cast<Rect2f>();
                    d.score = t[1].cast<float>();
                    d.label_id = t[2].cast<int32_t>();
                    return d;
                }));
        bind_ultralytics_det(m);
    }
}
