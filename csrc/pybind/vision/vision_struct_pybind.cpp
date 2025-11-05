//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "pybind/utils/result_convert.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    void bind_vision_struct(const pybind11::module& m) {
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
                            "Point2f pickle with invalid state!");
                    Point2f point2f;
                    point2f.x = t[0].cast<float>();
                    point2f.y = t[1].cast<float>();
                    return point2f;
                }))
            .def("to_dict", point2f_to_dict)
            .def_static("from_dict", dict_to_point2f)
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
                            "Point3f pickle with invalid state!");
                    Point3f point3f;
                    point3f.x = t[0].cast<float>();
                    point3f.y = t[1].cast<float>();
                    point3f.z = t[2].cast<float>();
                    return point3f;
                }))
            .def("to_dict", point3f_to_dict)
            .def_static("from_dict", dict_to_point3f)
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
                            "Rect2f pickle with invalid state!");
                    Rect2f rect2f;
                    rect2f.x = t[0].cast<float>();
                    rect2f.y = t[1].cast<float>();
                    rect2f.width = t[2].cast<float>();
                    rect2f.height = t[3].cast<float>();
                    return rect2f;
                }))
            .def("to_dict", rect2f_to_dict)
            .def_static("from_dict", dict_to_rect2f)
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
                            "RotatedRect pickle with invalid state!");
                    RotatedRect rotated_rect;
                    rotated_rect.xc = t[0].cast<float>();
                    rotated_rect.yc = t[1].cast<float>();
                    rotated_rect.width = t[2].cast<float>();
                    rotated_rect.height = t[3].cast<float>();
                    rotated_rect.angle = t[4].cast<float>();
                    return rotated_rect;
                }))
            .def("to_dict", rotated_rect_to_dict)
            .def_static("from_dict", dict_to_rotated_rect)
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
                }))
            .def("to_dict", mask_to_dict)
            .def_static("from_dict", dict_to_mask);


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
                }))
            .def("to_dict", [](const ClassifyResult& c) {
                pybind11::dict d;
                d["label_ids"] = pybind11::cast(c.label_ids);
                d["scores"] = pybind11::cast(c.scores);
                if (!c.feature.empty()) {
                    d["feature"] = pybind11::cast(c.feature);
                }
                return d;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                if (!d.contains("label_ids") || !d.contains("scores")) {
                    throw std::runtime_error("ClassifyResult.from_dict: missing required fields");
                }
                ClassifyResult c;
                c.label_ids = d["label_ids"].cast<std::vector<int32_t>>();
                c.scores = d["scores"].cast<std::vector<float>>();
                if (d.contains("feature")) {
                    c.feature = d["feature"].cast<std::vector<float>>();
                }
                return c;
            });


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
                }))
            .def("to_dict", [](const DetectionResult& d) {
                pybind11::dict result;
                auto box_dict = rect2f_to_dict(d.box);
                result["box"] = box_dict;
                result["score"] = d.score;
                result["label_id"] = d.label_id;
                return result;
            }).def_static("from_dict", [](const pybind11::dict& d) {
                DetectionResult r;
                // 解析嵌套的 box 字典
                if (d.contains("box")) {
                    const auto box_dict = d["box"].cast<pybind11::dict>();
                    r.box = dict_to_rect2f(box_dict);
                }
                if (d.contains("score")) r.score = d["score"].cast<float>();
                if (d.contains("label_id")) r.label_id = d["label_id"].cast<int32_t>();

                return r;
            }).def("__str__", [](const DetectionResult& d) {
                return "<DetectionResult label_id=" + std::to_string(d.label_id) +
                    ", score=" + std::to_string(d.score) + ", box=" + d.box.to_string() + ">";
            })
            .def("__repr__", [](const DetectionResult& d) {
                return "<DetectionResult label_id=" + std::to_string(d.label_id) +
                    ", score=" + std::to_string(d.score) + ", box=" + d.box.to_string() + ">";
            });

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
                }))
            .def("to_dict", [](const InstanceSegResult& r) {
                pybind11::dict d;
                d["box"] = rect2f_to_dict(r.box);
                d["mask"] = mask_to_dict(r.mask);
                d["score"] = r.score;
                d["label_id"] = r.label_id;
                return d;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                if (!d.contains("box") || !d.contains("mask") ||
                    !d.contains("score") || !d.contains("label_id")) {
                    throw std::runtime_error("InstanceSegResult.from_dict: missing required fields");
                }
                InstanceSegResult r;
                r.box = dict_to_rect2f(d["box"]);
                r.mask = dict_to_mask(d["mask"]);
                r.score = d["score"].cast<float>();
                r.label_id = d["label_id"].cast<int32_t>();
                return r;
            });


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
                }))
            .def("to_dict", [](const ObbResult& d) {
                pybind11::dict result;
                result["rotated_box"] = rotated_rect_to_dict(d.rotated_box);;
                result["score"] = d.score;
                result["label_id"] = d.label_id;
                return result;
            }).def_static("from_dict", [](const pybind11::dict& d) {
                ObbResult r;
                // 解析嵌套的 box 字典
                if (d.contains("rotated_box")) {
                    const auto box_dict = d["rotated_box"].cast<pybind11::dict>();
                    r.rotated_box = dict_to_rotated_rect(box_dict);
                }
                if (d.contains("score")) r.score = d["score"].cast<float>();
                if (d.contains("label_id")) r.label_id = d["label_id"].cast<int32_t>();
                return r;
            }).def("__str__", [](const ObbResult& d) {
                return "<ObbResult label_id=" + std::to_string(d.label_id) +
                    ", score=" + std::to_string(d.score) + ", box=" + d.rotated_box.to_string() + ">";
            })
            .def("__repr__", [](const ObbResult& d) {
                return "<ObbResult label_id=" + std::to_string(d.label_id) +
                    ", score=" + std::to_string(d.score) + ", box=" + d.rotated_box.to_string() + ">";
            });


        pybind11::class_<KeyPointsResult>(m, "KeyPointsResult")
            .def(pybind11::init())
            .def_readwrite("box", &KeyPointsResult::box)
            .def_readwrite("keypoints", &KeyPointsResult::keypoints)
            .def_readwrite("label_id", &KeyPointsResult::label_id)
            .def_readwrite("score", &KeyPointsResult::score)
            .def(pybind11::pickle(
                [](const KeyPointsResult& d) {
                    return pybind11::make_tuple(d.box, d.keypoints, d.score, d.label_id);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 4)
                        throw std::runtime_error("KeyPointsResult pickle with Invalid state!");
                    KeyPointsResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.keypoints = t[1].cast<std::vector<Point3f>>();
                    d.score = t[2].cast<float>();
                    d.label_id = t[3].cast<int32_t>();
                    return d;
                }))
            .def("to_dict", [](const KeyPointsResult& p) {
                pybind11::dict result;
                result["box"] = rect2f_to_dict(p.box);
                // 2. 处理 keypoints（Point3f 的 vector）
                pybind11::list kps;
                for (const auto& kp : p.keypoints) {
                    pybind11::dict pt;
                    pt["x"] = kp.x;
                    pt["y"] = kp.y;
                    pt["z"] = kp.z;
                    kps.append(pt);
                }
                result["keypoints"] = kps;
                // 3. 其他字段
                result["score"] = p.score;
                result["label_id"] = p.label_id;
                return result;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                KeyPointsResult r;
                // 1. box
                if (d.contains("box")) {
                    r.box = dict_to_rect2f(d["box"]);
                }
                // 2. keypoints
                if (d.contains("keypoints")) {
                    auto kps_list = d["keypoints"].cast<pybind11::list>();
                    for (auto kp_obj : kps_list) {
                        auto kp_dict = kp_obj.cast<pybind11::dict>();
                        Point3f kp;
                        kp.x = kp_dict["x"].cast<float>();
                        kp.y = kp_dict["y"].cast<float>();
                        kp.z = kp_dict["z"].cast<float>();
                        r.keypoints.push_back(kp);
                    }
                }
                // 3. 其他字段
                if (d.contains("score")) r.score = d["score"].cast<float>();
                if (d.contains("label_id")) r.label_id = d["label_id"].cast<int32_t>();
                return r;
            });

        pybind11::enum_<FaceAntiSpoofResult>(m, "FaceAntiSpoofResult")
            .value("REAL", FaceAntiSpoofResult::REAL)
            .value("FUZZY", FaceAntiSpoofResult::FUZZY)
            .value("SPOOF", FaceAntiSpoofResult::SPOOF);

        pybind11::class_<OCRResult>(m, "OCRResult")
            .def(pybind11::init())
            .def_readwrite("boxes", &OCRResult::boxes)
            .def_readwrite("text", &OCRResult::text)
            .def_readwrite("rec_scores", &OCRResult::rec_scores)
            .def_readwrite("cls_scores", &OCRResult::cls_scores)
            .def_readwrite("cls_labels", &OCRResult::cls_labels)
            .def_readwrite("table_boxes", &OCRResult::table_boxes)
            .def_readwrite("table_structure", &OCRResult::table_structure)
            .def_readwrite("table_html", &OCRResult::table_html)
            .def("to_dict", [](const OCRResult& r) {
                pybind11::dict d;
                d["boxes"] = r.boxes;
                d["text"] = r.text;
                d["rec_scores"] = r.rec_scores;
                d["cls_scores"] = r.cls_scores;
                d["cls_labels"] = r.cls_labels;
                d["table_boxes"] = r.table_boxes;
                d["table_structure"] = r.table_structure;
                d["table_html"] = r.table_html;
                return d;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                OCRResult r = {};
                if (d.contains("boxes")) r.boxes = d["boxes"].cast<std::vector<std::array<int, 8>>>();
                if (d.contains("text")) r.text = d["text"].cast<std::vector<std::string>>();
                if (d.contains("rec_scores")) r.rec_scores = d["rec_scores"].cast<std::vector<float>>();
                if (d.contains("cls_scores")) r.cls_scores = d["cls_scores"].cast<std::vector<float>>();
                if (d.contains("cls_labels")) r.cls_labels = d["cls_labels"].cast<std::vector<int32_t>>();
                if (d.contains("table_boxes")) r.table_boxes = d["table_boxes"].cast<std::vector<std::array<int, 8>>>();
                if (d.contains("table_structure"))
                    r.table_structure = d["table_structure"].cast<std::vector<
                        std::string>>();
                if (d.contains("table_html")) r.table_html = d["table_html"].cast<std::string>();
                return r;
            });


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
                }))
            .def("to_dict", [](const FaceRecognitionResult& f) {
                pybind11::dict result;
                result["embedding"] = pybind11::cast(f.embedding);
                return result;
            }).def_static("from_dict", [](const pybind11::dict& d) {
                FaceRecognitionResult f;
                // 解析嵌套的 box 字典
                if (d.contains("embedding")) {
                    f.embedding = d["embedding"].cast<std::vector<float>>();
                }
                return f;
            });

        pybind11::class_<LprResult>(m, "LprResult")
            .def(pybind11::init())
            .def_readwrite("box", &LprResult::box)
            .def_readwrite("landmarks", &LprResult::keypoints)
            .def_readwrite("label_id", &LprResult::label_id)
            .def_readwrite("score", &LprResult::score)
            .def_readwrite("car_plate_str", &LprResult::car_plate_str)
            .def_readwrite("car_plate_color", &LprResult::car_plate_color)
            .def(pybind11::pickle(
                [](const LprResult& d) {
                    return pybind11::make_tuple(d.box, d.keypoints, d.label_id, d.score, d.car_plate_str,
                                                d.car_plate_color);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 6)
                        throw std::runtime_error("LprResult pickle with Invalid state!");
                    LprResult d;
                    d.box = t[0].cast<Rect2f>();
                    d.keypoints = t[1].cast<std::vector<Point3f>>();
                    d.label_id = t[2].cast<int32_t>();
                    d.score = t[3].cast<float>();
                    d.car_plate_str = t[4].cast<std::string>();
                    d.car_plate_color = t[5].cast<std::string>();
                    return d;
                }))
            .def("to_dict", [](const LprResult& p) {
                pybind11::dict result;
                result["box"] = rect2f_to_dict(p.box);
                // 2. 处理 keypoints（Point3f 的 vector）
                pybind11::list kps;
                for (const auto& kp : p.keypoints) {
                    pybind11::dict pt;
                    pt["x"] = kp.x;
                    pt["y"] = kp.y;
                    pt["z"] = kp.z;
                    kps.append(pt);
                }
                result["keypoints"] = kps;
                // 3. 其他字段
                result["score"] = p.score;
                result["label_id"] = p.label_id;
                result["car_plate_str"] = p.car_plate_str;
                result["car_plate_color"] = p.car_plate_color;
                return result;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                LprResult r;
                // 1. box
                if (d.contains("box")) {
                    r.box = dict_to_rect2f(d["box"]);
                }
                // 2.landmarks
                if (d.contains("landmarks")) {
                    auto kps_list = d["landmarks"].cast<pybind11::list>();
                    for (auto kp_obj : kps_list) {
                        auto kp_dict = kp_obj.cast<pybind11::dict>();
                        Point3f lmk;
                        lmk.x = kp_dict["x"].cast<float>();
                        lmk.y = kp_dict["y"].cast<float>();
                        lmk.z = kp_dict["z"].cast<float>();
                        r.keypoints.push_back(lmk);
                    }
                }
                // 3. 其他字段
                if (d.contains("score")) r.score = d["score"].cast<float>();
                if (d.contains("label_id")) r.label_id = d["label_id"].cast<int32_t>();
                if (d.contains("car_plate_str")) r.car_plate_str = d["car_plate_str"].cast<std::string>();
                if (d.contains("car_plate_color")) r.car_plate_color = d["car_plate_color"].cast<std::string>();
                return r;
            });

        pybind11::class_<LetterBoxRecord>(m, "LetterBoxRecord", pybind11::dynamic_attr())
            .def(pybind11::init())
            .def_readwrite("ipt_w", &LetterBoxRecord::ipt_w)
            .def_readwrite("ipt_h", &LetterBoxRecord::ipt_h)
            .def_readwrite("out_w", &LetterBoxRecord::out_w)
            .def_readwrite("out_h", &LetterBoxRecord::out_h)
            .def_readwrite("pad_w", &LetterBoxRecord::pad_w)
            .def_readwrite("pad_h", &LetterBoxRecord::pad_h)
            .def_readwrite("scale", &LetterBoxRecord::scale)
            .def(pybind11::pickle(
                [](const LetterBoxRecord& l) {
                    return pybind11::make_tuple(l.ipt_w, l.ipt_h, l.out_w, l.out_h, l.pad_w, l.pad_h, l.scale);
                },
                [](const pybind11::tuple& t) {
                    if (t.size() != 7)
                        throw std::runtime_error("LetterBoxRecord pickle with Invalid state!");
                    const LetterBoxRecord l{
                        t[0].cast<float>(),
                        t[1].cast<float>(),
                        t[2].cast<float>(),
                        t[3].cast<float>(),
                        t[4].cast<float>(),
                        t[5].cast<float>(),
                        t[6].cast<float>(),
                    };
                    return l;
                }))
            .def("to_dict", [](const LetterBoxRecord& l) {
                pybind11::dict d;
                d["ipt_w"] = l.ipt_w;
                d["ipt_h"] = l.ipt_h;
                d["out_w"] = l.out_w;
                d["out_h"] = l.out_h;
                d["pad_w"] = l.pad_w;
                d["pad_h"] = l.pad_h;
                d["scale"] = l.scale;
                return d;
            })
            .def_static("from_dict", [](const pybind11::dict& d) {
                LetterBoxRecord l = {};
                if (d.contains("ipt_w")) l.ipt_w = d["ipt_w"].cast<float>();
                if (d.contains("ipt_h")) l.ipt_h = d["ipt_h"].cast<float>();
                if (d.contains("out_w")) l.out_w = d["out_w"].cast<float>();
                if (d.contains("out_h")) l.out_h = d["out_h"].cast<float>();
                if (d.contains("pad_w")) l.pad_w = d["pad_w"].cast<float>();
                if (d.contains("pad_h")) l.pad_h = d["pad_h"].cast<float>();
                if (d.contains("scale")) l.scale = d["scale"].cast<float>();
                return l;
            })
            .def("__str__", &LetterBoxRecord::to_string)
            .def("__repr__", &LetterBoxRecord::to_string);
    }
} // namespace modeldeploy
