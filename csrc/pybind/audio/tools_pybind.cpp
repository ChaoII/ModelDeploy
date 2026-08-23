#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio/tools/resampler.h"
#include "audio/tools/fbank.h"
#include "audio/tools/waveform.h"

namespace modeldeploy {
namespace audio {

void bind_tools(pybind11::module& m) {
    pybind11::class_<tool::Resampler>(m, "Resampler")
        .def_static("resample", &tool::Resampler::resample);
    pybind11::class_<tool::Fbank>(m, "Fbank")
        .def(pybind11::init<int, int>(), pybind11::arg("sample_rate") = 16000, pybind11::arg("num_bins") = 80)
        .def("compute", &tool::Fbank::compute);
    pybind11::class_<tool::Spectrum>(m, "Spectrum")
        .def(pybind11::init<int>(), pybind11::arg("fft_n") = 1024)
        .def("magnitudes", &tool::Spectrum::magnitudes);
}

} // namespace audio
} // namespace modeldeploy
