#include <catch2/catch_test_macros.hpp>
#include "csrc/vision/ocr/doc_to_markdown.h"

using namespace modeldeploy::vision::ocr;

// Constant-pass: readiness semantics without any models/layout
TEST_CASE("DocToMarkdown ready semantics", "[ocr][doc]") {
    DocToMarkdown d;
    REQUIRE_FALSE(d.ready());          // nothing set
}
