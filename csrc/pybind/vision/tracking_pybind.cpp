//
// Created for the MOT tracking family Python bindings.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vision/tracking/bytetrack.h"
#include "vision/tracking/botsort.h"
#include "vision/tracking/strongsort.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy::vision {
    void bind_tracking(const pybind11::module& m) {
        using namespace tracking;

        // TrackState is bound as a plain int-backed enum (TrackResult.state is int).
        pybind11::enum_<TrackState>(m, "TrackState")
            .value("New", TrackState::New)
            .value("Tracked", TrackState::Tracked)
            .value("Lost", TrackState::Lost)
            .value("Removed", TrackState::Removed);

        // Rect2f is already bound in vision_struct_pybind.cpp; Detection::box /
        // TrackResult::box reference that existing type (no re-binding here).
        pybind11::class_<Detection>(m, "Detection")
            .def(pybind11::init<>())
            .def_readwrite("box", &Detection::box)
            .def_readwrite("score", &Detection::score)
            .def_readwrite("label_id", &Detection::label_id)
            .def_readwrite("feature", &Detection::feature)
            .def("__repr__", [](const Detection& d) {
                return "<Detection label_id=" + std::to_string(d.label_id) +
                    ", score=" + std::to_string(d.score) +
                    ", box=(" + d.box.to_string() + ")>";
            });

        pybind11::class_<TrackResult>(m, "TrackResult")
            .def(pybind11::init<>())
            .def_readwrite("track_id", &TrackResult::track_id, pybind11::return_value_policy::copy)
            .def_readwrite("box", &TrackResult::box)
            .def_readwrite("score", &TrackResult::score)
            .def_readwrite("label_id", &TrackResult::label_id)
            .def_readwrite("state", &TrackResult::state)
            .def_readwrite("feature", &TrackResult::feature)
            .def("__repr__", [](const TrackResult& t) {
                return "<TrackResult track_id=" + std::to_string(t.track_id) +
                    ", state=" + std::to_string(t.state) +
                    ", box=(" + t.box.to_string() + ")>";
            });

        pybind11::class_<ByteTracker>(m, "ByteTracker")
            .def(pybind11::init<>())
            .def("set_params", &ByteTracker::set_params,
                 pybind11::arg("track_thresh") = 0.5f,
                 pybind11::arg("high_thresh") = 0.5f,
                 pybind11::arg("low_thresh") = 0.1f,
                 pybind11::arg("max_age") = 30,
                 pybind11::arg("min_hits") = 3,
                 pybind11::arg("iou_threshold") = 0.3f)
            .def("update", &ByteTracker::update,
                 pybind11::arg("detections"),
                 pybind11::arg("frame") = pybind11::none(),
                 pybind11::arg("timestamp") = -1.0)
            .def("reset", &ByteTracker::reset);

        pybind11::class_<BotSortTracker>(m, "BotSortTracker")
            .def(pybind11::init<>())
            .def("set_params", &BotSortTracker::set_params,
                 pybind11::arg("track_thresh") = 0.5f,
                 pybind11::arg("high_thresh") = 0.5f,
                 pybind11::arg("low_thresh") = 0.1f,
                 pybind11::arg("max_age") = 30,
                 pybind11::arg("min_hits") = 3,
                 pybind11::arg("iou_threshold") = 0.3f,
                 pybind11::arg("match_thresh") = 0.8f,
                 pybind11::arg("fuse_score_weight") = 0.5f,
                 pybind11::arg("ema_alpha") = 0.9f,
                 pybind11::arg("with_cmc") = true)
            .def("set_reid", &BotSortTracker::set_reid)
            .def("update", &BotSortTracker::update,
                 pybind11::arg("detections"),
                 pybind11::arg("frame") = pybind11::none(),
                 pybind11::arg("timestamp") = -1.0)
            .def("reset", &BotSortTracker::reset);

        pybind11::class_<StrongSortTracker>(m, "StrongSortTracker")
            .def(pybind11::init<>())
            .def("set_params", &StrongSortTracker::set_params,
                 pybind11::arg("track_thresh") = 0.5f,
                 pybind11::arg("high_thresh") = 0.5f,
                 pybind11::arg("low_thresh") = 0.1f,
                 pybind11::arg("max_age") = 30,
                 pybind11::arg("min_hits") = 3,
                 pybind11::arg("iou_threshold") = 0.3f,
                 pybind11::arg("match_thresh") = 0.8f,
                 pybind11::arg("ema_alpha") = 0.9f,
                 pybind11::arg("appearance_priority") = 0.7f,
                 pybind11::arg("with_cmc") = true)
            .def("set_reid", &StrongSortTracker::set_reid)
            .def("update", &StrongSortTracker::update,
                 pybind11::arg("detections"),
                 pybind11::arg("frame") = pybind11::none(),
                 pybind11::arg("timestamp") = -1.0)
            .def("reset", &StrongSortTracker::reset);
    }
} // namespace modeldeploy
