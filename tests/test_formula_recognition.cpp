#include <catch2/catch_test_macros.hpp>
#include "csrc/vision/ocr/formula_recognition.h"

using namespace modeldeploy::vision::ocr;

// Constant-pass: ctor error path / is_initialized false without weights
TEST_CASE("FormulaRecognizer construction", "[formula]") {
    FormulaRecognizer m("nonexistent_formula.onnx");
    REQUIRE_FALSE(m.is_initialized());
}
