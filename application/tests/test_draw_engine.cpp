#include <catch2/catch_test_macros.hpp>
#include "draw_engine.hpp"
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy::vision;

TEST_CASE("DrawEngine construct", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
}

TEST_CASE("DrawEngine draw on empty results", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
    ImageData img(100, 100, MdImageType::PKG_BGR_U8);
    std::vector<InferResult> results;
    REQUIRE_NOTHROW(de.draw(img, results));
}

TEST_CASE("DrawEngine draw with detection results", "[draw]") {
    DrawConfig cfg;
    cfg.show_label = true;
    cfg.show_score = true;
    DrawEngine de(cfg);
    ImageData img(200, 200, MdImageType::PKG_BGR_U8);

    InferResult r;
    r.model_name = "det";
    r.type = "detection";
    DetectionBox box;
    box.x = 10; box.y = 10; box.w = 50; box.h = 50;
    box.score = 0.95f;
    box.label_id = 1;
    r.boxes.push_back(box);

    std::vector<InferResult> results = {r};
    REQUIRE_NOTHROW(de.draw(img, results));

    // 确认像素被修改了
    auto* data = img.data();
    bool has_nonzero = false;
    size_t total = static_cast<size_t>(200) * 200 * 3;
    for (size_t i = 0; i < total; ++i) {
        if (data[i] != 0) { has_nonzero = true; break; }
    }
    REQUIRE(has_nonzero);
}
