#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vision/tools/detections.h"
#include "vision/tools/zone.h"
#include "vision/tools/metrics.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy {
namespace vision {

namespace {
Rect2f r2f(const pybind11::handle& r) {
    if (pybind11::isinstance<Rect2f>(r)) return r.cast<Rect2f>();
    auto t = r.cast<pybind11::tuple>();
    return Rect2f(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>());
}

Point2f pt2f(const pybind11::handle& p) {
    if (pybind11::isinstance<Point2f>(p)) return p.cast<Point2f>();
    auto t = p.cast<pybind11::tuple>();
    return Point2f(t[0].cast<float>(), t[1].cast<float>());
}

tracking::TrackResult tr(const pybind11::handle& d) {
    auto dict = d.cast<pybind11::dict>();
    tracking::TrackResult r;
    r.track_id = dict.contains("track_id") ? dict["track_id"].cast<int>() : -1;
    r.score = dict.contains("score") ? dict["score"].cast<float>() : 0.0f;
    r.label_id = dict.contains("label_id") ? dict["label_id"].cast<int>() : 0;
    r.box = r2f(dict["box"]);
    return r;
}

std::vector<tracking::TrackResult> trs(const pybind11::iterable& it) {
    std::vector<tracking::TrackResult> out;
    for (auto item : it) out.push_back(tr(item));
    return out;
}
} // namespace

void bind_tools(pybind11::module& m) {
    using namespace modeldeploy::vision::tool;
    pybind11::class_<Detections>(m, "Detections")
        .def(pybind11::init<>())
        .def_readwrite("boxes", &Detections::boxes)
        .def_readwrite("class_id", &Detections::class_id)
        .def_readwrite("confidence", &Detections::confidence)
        .def_readwrite("tracker_id", &Detections::tracker_id)
        .def("__len__", &Detections::size);
    m.def("iou", [](const pybind11::object& a, const pybind11::object& b) { return iou(r2f(a), r2f(b)); });
    m.def("nms", [](Detections& d, float t) { nms(d, t); }, pybind11::arg("d"), pybind11::arg("iou_threshold") = 0.5f);
    m.def("from_track", [](const pybind11::iterable& t) { return from_track(trs(t)); });
    pybind11::class_<LineZone>(m, "LineZone")
        .def(pybind11::init([](const pybind11::object& a, const pybind11::object& b) { return LineZone(pt2f(a), pt2f(b)); }))
        .def("trigger", [](LineZone& z, const pybind11::handle& p) { return z.trigger(pt2f(p)); })
        .def("trigger_count", &LineZone::trigger_count);
    pybind11::class_<PolygonZone>(m, "PolygonZone")
        .def(pybind11::init([](const pybind11::iterable& pts) {
            std::vector<Point2f> v;
            for (auto p : pts) v.push_back(pt2f(p));
            return PolygonZone(v);
        }))
        .def("contains", [](PolygonZone& z, const pybind11::handle& p) { return z.contains(pt2f(p)); })
        .def("current_count", &PolygonZone::current_count);
    m.def("evaluate_metrics", &evaluate_metrics);
}

} // namespace vision
} // namespace modeldeploy
