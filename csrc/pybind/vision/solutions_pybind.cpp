#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/distance_estimator.h"
#include "vision/solutions/workout_monitor.h"
#include "vision/solutions/parking_manager.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy {
namespace vision {

namespace {
// 兼容 Python tuple：把 (x, y) 转成 Point2f
Point2f p2f(const pybind11::handle& p) {
    if (pybind11::isinstance<Point2f>(p)) return p.cast<Point2f>();
    auto t = p.cast<pybind11::tuple>();
    return Point2f(t[0].cast<float>(), t[1].cast<float>());
}

// 兼容 Python dict：{track_id, box, score, label_id} 转成 TrackResult
tracking::TrackResult tr(const pybind11::handle& d) {
    auto dict = d.cast<pybind11::dict>();
    tracking::TrackResult r;
    r.track_id = dict.contains("track_id") ? dict["track_id"].cast<int>() : -1;
    r.score = dict.contains("score") ? dict["score"].cast<float>() : 0.0f;
    r.label_id = dict.contains("label_id") ? dict["label_id"].cast<int>() : 0;
    auto box = dict["box"];
    if (pybind11::isinstance<Rect2f>(box)) {
        r.box = box.cast<Rect2f>();
    } else {
        auto b = box.cast<pybind11::tuple>();
        r.box = Rect2f(b[0].cast<float>(), b[1].cast<float>(), b[2].cast<float>(), b[3].cast<float>());
    }
    return r;
}

std::vector<tracking::TrackResult> trs(const pybind11::iterable& it) {
    std::vector<tracking::TrackResult> out;
    for (auto item : it) out.push_back(tr(item));
    return out;
}
} // namespace

void bind_solutions(pybind11::module& m) {
    using namespace modeldeploy::vision::solution;
    pybind11::class_<ObjectCounter>(m, "ObjectCounter")
        .def(pybind11::init<>())
        .def("set_line", [](ObjectCounter& s, const pybind11::object& a, const pybind11::object& b) {
            s.set_line(p2f(a), p2f(b));
        })
        .def("set_region", [](ObjectCounter& s, const pybind11::iterable& pts) {
            std::vector<Point2f> v;
            for (auto p : pts) v.push_back(p2f(p));
            s.set_region(v);
        })
        .def("set_classes", &ObjectCounter::set_classes)
        .def("update", [](ObjectCounter& s, const pybind11::iterable& t) { s.update(trs(t)); })
        .def("line_in", [](ObjectCounter& s) { return s.stats().line_in; })
        .def("line_out", [](ObjectCounter& s) { return s.stats().line_out; })
        .def("class_count", [](ObjectCounter& s) { return s.stats().class_count; })
        .def("region_count", &ObjectCounter::region_count);
    pybind11::class_<Heatmap>(m, "Heatmap")
        .def(pybind11::init<>())
        .def("set_size", &Heatmap::set_size)
        .def("update", [](Heatmap& s, const pybind11::iterable& t, int w, int h) { s.update(trs(t), w, h); })
        .def("peak", &Heatmap::peak)
        .def("heat_at", &Heatmap::heat_at);
    pybind11::class_<SpeedEstimator>(m, "SpeedEstimator")
        .def(pybind11::init<>())
        .def("set_meter_per_pixel", &SpeedEstimator::set_meter_per_pixel)
        .def("update", [](SpeedEstimator& s, const pybind11::iterable& t, double ts) { s.update(trs(t), ts); })
        .def("speeds_m_s", &SpeedEstimator::speeds_m_s);
    pybind11::class_<DistanceEstimator>(m, "DistanceEstimator")
        .def(pybind11::init<>())
        .def("set_meter_per_pixel", &DistanceEstimator::set_meter_per_pixel)
        .def("pair_distances_m", [](DistanceEstimator& s, const pybind11::iterable& t) { return s.pair_distances_m(trs(t)); });
    pybind11::class_<WorkoutMonitor>(m, "WorkoutMonitor")
        .def(pybind11::init<float, float>(), pybind11::arg("min_deg") = 70.0f, pybind11::arg("max_deg") = 160.0f)
        .def_static("angle", [](const Point3f& a, const Point3f& b, const Point3f& c) {
            return WorkoutMonitor::angle(a, b, c);
        })
        .def("update", &WorkoutMonitor::update)
        .def("reps", &WorkoutMonitor::reps);
    pybind11::class_<ParkingManager>(m, "ParkingManager")
        .def(pybind11::init<>())
        .def("set_slots", [](ParkingManager& s, const pybind11::iterable& slots) {
            std::vector<std::vector<Point2f>> v;
            for (auto slot : slots) {
                std::vector<Point2f> p;
                for (auto pt : slot.cast<pybind11::iterable>()) p.push_back(p2f(pt));
                v.push_back(p);
            }
            s.set_slots(v);
        })
        .def("update", [](ParkingManager& s, const pybind11::iterable& t) { s.update(trs(t)); })
        .def("occupancy", &ParkingManager::occupancy);
}

} // namespace vision
} // namespace modeldeploy
