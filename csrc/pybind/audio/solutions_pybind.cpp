#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include "audio/solutions/speaker_search.h"
#include "audio/solutions/tts_batcher.h"

namespace modeldeploy {
namespace audio {

void bind_solutions(pybind11::module& m) {
    pybind11::class_<solution::SpeakerSearch>(m, "SpeakerSearch")
        .def(pybind11::init<>())
        .def("enroll", &solution::SpeakerSearch::enroll)
        .def("match", &solution::SpeakerSearch::match, pybind11::arg("embedding"), pybind11::arg("k") = 1);
    pybind11::class_<solution::TTSBatcher>(m, "TTSBatcher")
        .def(pybind11::init<>())
        .def("enqueue", &solution::TTSBatcher::enqueue)
        .def("dequeue_all", &solution::TTSBatcher::dequeue_all);
}

} // namespace audio
} // namespace modeldeploy
