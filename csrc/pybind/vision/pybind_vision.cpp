//
// Created by aichao on 2025/6/9.
//

#include <csrc/vision/common/display/display.h>

#include "csrc/vision/common/result.h"
#include <pybind11/pybind11.h>

namespace modeldeploy::vision {
    void bind_ultralytics_cls(pybind11::module&);
    void bind_ultralytics_det(pybind11::module&);
    void bind_ultralytics_iseg(pybind11::module&);
    void bind_ultralytics_obb(pybind11::module&);
    void bind_ultralytics_pose(pybind11::module&);
    void bind_lpr_det(pybind11::module&);
    void bind_lpr_rec(pybind11::module&);

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

        pybind11::class_<InstanceSegResult>(m, "InstanceSegResult")
            .def(pybind11::init())
            .def_readwrite("box", &InstanceSegResult::box)
            .def_readwrite("mask", &InstanceSegResult::mask)
            .def_readwrite("label_id", &InstanceSegResult::label_id)
            .def_readwrite("score", &InstanceSegResult::score)
            .def(pybind11::pickle(
                [](const InstanceSegResult& d) {
                    return pybind11::make_tuple(d.box, d.mask, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 4)
                        throw std::runtime_error("InstanceSegResult pickle with Invalid state!");
                    InstanceSegResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.mask = t[1].cast<Mask>();
                    d.score = t[2].cast<float>();
                    d.label_id = t[3].cast<int32_t>();
                    return d;
                }));


        pybind11::class_<ObbResult>(m, "ObbResult")
            .def(pybind11::init())
            .def_readwrite("rotated_box", &ObbResult::rotated_box)
            .def_readwrite("score", &ObbResult::score)
            .def_readwrite("label_id", &ObbResult::label_id)
            .def(pybind11::pickle(
                [](const ObbResult& d) {
                    return pybind11::make_tuple(d.rotated_box, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 3)
                        throw std::runtime_error("ObbResult pickle with Invalid state!");
                    ObbResult d;
                    d.rotated_box = t[0].cast<RotatedRect>();
                    d.score = t[1].cast<float>();
                    d.label_id = t[2].cast<int32_t>();
                    return d;
                }));


        pybind11::class_<PoseResult>(m, "PoseResult")
            .def(pybind11::init())
            .def_readwrite("box", &PoseResult::box)
            .def_readwrite("mask", &PoseResult::keypoints)
            .def_readwrite("label_id", &PoseResult::label_id)
            .def_readwrite("score", &PoseResult::score)
            .def(pybind11::pickle(
                [](const PoseResult& d) {
                    return pybind11::make_tuple(d.box, d.keypoints, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 4)
                        throw std::runtime_error("PoseResult pickle with Invalid state!");
                    PoseResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.keypoints = t[1].cast<std::vector<Point3f>>();
                    d.score = t[2].cast<float>();
                    d.label_id = t[3].cast<int32_t>();
                    return d;
                }));


        pybind11::class_<OCRResult>(m, "OCRResult")
            .def(pybind11::init())
            .def_readwrite("boxes", &OCRResult::boxes)
            .def_readwrite("text", &OCRResult::text)
            .def_readwrite("rec_scores", &OCRResult::rec_scores)
            .def_readwrite("cls_scores", &OCRResult::cls_scores)
            .def_readwrite("cls_labels", &OCRResult::cls_labels)
            .def_readwrite("table_boxes", &OCRResult::table_boxes)
            .def_readwrite("table_structure", &OCRResult::table_structure)
            .def_readwrite("table_html", &OCRResult::table_html);

        pybind11::class_<DetectionLandmarkResult>(m, "DetectionLandmarkResult")
            .def(pybind11::init())
            .def_readwrite("box", &DetectionLandmarkResult::box)
            .def_readwrite("landmarks", &DetectionLandmarkResult::landmarks)
            .def_readwrite("label_id", &DetectionLandmarkResult::label_id)
            .def_readwrite("score", &DetectionLandmarkResult::score).def(pybind11::pickle(
                [](const DetectionLandmarkResult& d) {
                    return pybind11::make_tuple(d.box, d.landmarks, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 4)
                        throw std::runtime_error("DetectionLandmarkResult pickle with Invalid state!");
                    DetectionLandmarkResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.landmarks = t[1].cast<std::vector<Point2f>>();
                    d.score = t[2].cast<float>();
                    d.label_id = t[3].cast<int32_t>();
                    return d;
                }));


        pybind11::class_<FaceRecognitionResult>(m, "FaceRecognitionResult")
            .def(pybind11::init())
            .def_readwrite("embedding", &FaceRecognitionResult::embedding)
            .def(pybind11::pickle(
                [](const FaceRecognitionResult& d) {
                    return pybind11::make_tuple(d.embedding);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 1)
                        throw std::runtime_error("FaceRecognitionResult pickle with Invalid state!");
                    FaceRecognitionResult d;
                    d.embedding = t[0].cast<std::vector<float>>();
                    return d;
                }));

        pybind11::class_<LprResult>(m, "LprResult")
            .def(pybind11::init())
            .def_readwrite("box", &LprResult::box)
            .def_readwrite("landmarks", &LprResult::landmarks)
            .def_readwrite("label_id", &LprResult::label_id)
            .def_readwrite("score", &LprResult::score)
            .def_readwrite("car_plate_str", &LprResult::car_plate_str)
            .def_readwrite("car_plate_color", &LprResult::car_plate_color)
            .def(pybind11::pickle(
                [](const LprResult& d) {
                    return pybind11::make_tuple(d.box, d.landmarks, d.label_id, d.score, d.car_plate_str,
                                                d.car_plate_color);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 6)
                        throw std::runtime_error("LprResult pickle with Invalid state!");
                    LprResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.landmarks = t[1].cast<std::vector<Point2f>>();
                    d.label_id = t[2].cast<int32_t>();
                    d.score = t[3].cast<float>();
                    d.car_plate_str = t[4].cast<std::string>();
                    d.car_plate_color = t[5].cast<std::string>();
                    return d;
                }));

        bind_ultralytics_cls(m);
        bind_ultralytics_det(m);
        bind_ultralytics_iseg(m);
        bind_ultralytics_obb(m);
        bind_ultralytics_pose(m);
        bind_lpr_det(m);
        bind_lpr_rec(m);
    }
}
