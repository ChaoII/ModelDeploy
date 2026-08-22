#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vision/barcode/barcode.h"
#include "vision/barcode/result.h"

namespace modeldeploy::vision {
    void bind_barcode(const pybind11::module& m) {
        using namespace barcode;
        m.attr("FMT_QR_CODE") = FMT_QR_CODE;
        m.attr("FMT_DATA_MATRIX") = FMT_DATA_MATRIX;
        m.attr("FMT_AZTEC") = FMT_AZTEC;
        m.attr("FMT_EAN_8") = FMT_EAN_8;
        m.attr("FMT_EAN_13") = FMT_EAN_13;
        m.attr("FMT_UPC_A") = FMT_UPC_A;
        m.attr("FMT_UPC_E") = FMT_UPC_E;
        m.attr("FMT_CODE_128") = FMT_CODE_128;
        m.attr("FMT_CODE_39") = FMT_CODE_39;
        m.attr("FMT_CODE_93") = FMT_CODE_93;
        m.attr("FMT_ITF") = FMT_ITF;
        m.attr("FMT_CODABAR") = FMT_CODABAR;
        m.attr("FMT_ALL") = FMT_ALL;

        pybind11::class_<BarcodeResult>(m, "BarcodeResult")
            .def(pybind11::init<>())
            .def_readwrite("text", &BarcodeResult::text)
            .def_readwrite("format", &BarcodeResult::format)
            .def_readwrite("quad", &BarcodeResult::quad)
            .def_readwrite("score", &BarcodeResult::score)
            .def_readwrite("is_qr", &BarcodeResult::is_qr)
            .def("__repr__", [](const BarcodeResult& r) {
                return "BarcodeResult(text=" + r.text + ", format=" + r.format
                       + ", is_qr=" + std::to_string(r.is_qr) + ")";
            });

        pybind11::class_<BarcodeDetector>(m, "BarcodeDetector")
            .def(pybind11::init<>())
            .def("set_formats", &BarcodeDetector::set_formats,
                 pybind11::arg("formats"))
            .def("detect", &BarcodeDetector::detect, pybind11::arg("img"))
            .def_property_readonly("formats", &BarcodeDetector::formats);
    }
}
