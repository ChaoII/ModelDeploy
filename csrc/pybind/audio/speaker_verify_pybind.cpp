#include "pybind/utils/utils.h"
#include "audio/speaker_verify/ecapa.h"
#include "audio/speaker_gallery.h"

namespace modeldeploy::audio {
    void bind_speaker_verify(pybind11::module& m) {
        pybind11::class_<speaker_verify::SpeakerVerify, BaseModel>(m, "SpeakerVerify")
            .def(pybind11::init([](const std::filesystem::path& model_file,
                                   const RuntimeOption& option) {
                     return std::make_unique<speaker_verify::SpeakerVerify>(
                         model_file.string(), option);
                 }),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](speaker_verify::SpeakerVerify& self, const std::vector<float>& samples) {
                     std::vector<float> emb;
                     self.predict(samples, &emb);
                     return emb;
                 }, pybind11::arg("samples"))
            .def("is_initialized", &speaker_verify::SpeakerVerify::is_initialized);

        pybind11::class_<SpeakerGallery>(m, "SpeakerGallery")
            .def(pybind11::init<>())
            .def("enroll", &SpeakerGallery::enroll, pybind11::arg("label"), pybind11::arg("embedding"))
            .def("remove", &SpeakerGallery::remove, pybind11::arg("label"))
            .def("match", &SpeakerGallery::match, pybind11::arg("embedding"), pybind11::arg("k") = 1)
            .def("size", &SpeakerGallery::size)
            .def("clear", &SpeakerGallery::clear);
    }
} // modeldeploy::audio
