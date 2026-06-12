#include <catch2/catch_test_macros.hpp>
#include "draw_engine.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

TEST_CASE("DrawEngine construct", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
    // Should not crash
}

TEST_CASE("DrawEngine draw on empty image", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<InferResult> results;
    REQUIRE_NOTHROW(de.draw(img, results));
}

TEST_CASE("DrawEngine draw with results", "[draw]") {
    DrawConfig cfg;
    cfg.show_label = true;
    cfg.show_score = true;
    DrawEngine de(cfg);
    cv::Mat img(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));

    InferResult r;
    r.model_name = "det";
    DetectionBox box;
    box.x = 10; box.y = 10; box.w = 50; box.h = 50;
    box.score = 0.95f;
    box.label_id = 1;
    r.boxes.push_back(box);

    std::vector<InferResult> results = {r};
    REQUIRE_NOTHROW(de.draw(img, results));

    // 确认像素被修改了（矩形区域内应该非零）
    auto* data = img.ptr();
    bool has_nonzero = false;
    for (int i = 0; i < 200 * 200 * 3; ++i) {
        if (data[i] != 0) { has_nonzero = true; break; }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("DrawEngine get_color consistency", "[draw]") {
    auto c1 = DrawEngine::get_color(0);
    auto c2 = DrawEngine::get_color(0);
    REQUIRE(c1[0] == c2[0]);
    REQUIRE(c1[1] == c2[1]);
    REQUIRE(c1[2] == c2[2]);
}

TEST_CASE("DrawEngine get_color different ids", "[draw]") {
    auto c1 = DrawEngine::get_color(0);
    auto c2 = DrawEngine::get_color(1);
    // Different IDs should (usually) give different colors
    bool same = (c1[0] == c2[0] && c1[1] == c2[1] && c1[2] == c2[2]);
    REQUIRE_FALSE(same);
}
