#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include "vision/common/result.h"
#include "vision/common/struct.h"
#include "vision/solutions/fall_detector.h"

using namespace modeldeploy::vision;
using namespace modeldeploy::vision::solution;

namespace {
KeyPointsResult make_person(const Point3f& sh_l, const Point3f& sh_r,
                            const Point3f& hp_l, const Point3f& hp_r) {
    KeyPointsResult person;
    person.box = Rect2f(0, 0, 100, 200);
    person.keypoints.assign(17, Point3f{0, 0, 1.f});
    person.keypoints[kLeftShoulder] = sh_l;
    person.keypoints[kRightShoulder] = sh_r;
    person.keypoints[kLeftHip] = hp_l;
    person.keypoints[kRightHip] = hp_r;
    person.score = 0.9f;
    return person;
}
} // namespace

TEST_CASE("angle(): 三点共线返回 180(伸直)", "[solution][fall]") {
    FallDetector fd;
    Point3f a{0, 100, 1}, b{0, 50, 1}, c{0, 0, 1};
    REQUIRE(fd.angle(a, b, c) == Catch::Approx(180.0f).margin(1e-2f));
}

TEST_CASE("angle(): 竖直站立躯干倾角≈0", "[solution][fall]") {
    FallDetector fd;
    // 髋在肩正下方 → 躯干与竖直方向夹角≈0
    Point3f hip{0, 200, 1}, sh{0, 100, 1}, vref{0, 200, 1};
    REQUIRE(fd.angle(hip, sh, vref) == Catch::Approx(0.0f).margin(1e-2f));
}

TEST_CASE("angle(): 平躺躯干倾角≈90", "[solution][fall]") {
    FallDetector fd;
    // 肩/髋同高、水平错开 → 躯干水平，相对竖直方向≈90
    Point3f hip{50, 100, 1}, sh{0, 100, 1}, vref{0, 200, 1};
    REQUIRE(fd.angle(hip, sh, vref) == Catch::Approx(90.0f).margin(1.0f));
}

TEST_CASE("update(): 站立→倒地状态推进", "[solution][fall]") {
    FallDetector fd;

    // 站立帧：肩 y=100，髋 y=200（髋在肩下方），倾角≈0 → Standing
    auto person = make_person({30, 100, 1}, {70, 100, 1}, {30, 200, 1}, {70, 200, 1});
    auto r = fd.update({person});
    REQUIRE(r.state == FallState::Standing);

    // 倒地帧（连续 >=3）：肩/髋齐平（y=100），倾角≈90 → 先 PreFall 再 Fallen
    auto fallen_person = make_person({30, 100, 1}, {70, 100, 1}, {60, 100, 1}, {40, 100, 1});
    r = fd.update({fallen_person});
    REQUIRE(r.state == FallState::PreFall);
    r = fd.update({fallen_person});
    r = fd.update({fallen_person});
    REQUIRE(r.state == FallState::Fallen);
    REQUIRE(r.confidence > 0.5f);

    // 复位：回到站立姿势，倾角 <30° → Standing
    r = fd.update({person});
    REQUIRE(r.state == FallState::Standing);
}

TEST_CASE("update(): reset() 复位状态", "[solution][fall]") {
    FallDetector fd;
    auto fallen_person = make_person({30, 100, 1}, {70, 100, 1}, {60, 100, 1}, {40, 100, 1});
    for (int i = 0; i < 3; ++i) fd.update({fallen_person});
    REQUIRE(fd.update({fallen_person}).state == FallState::Fallen);
    fd.reset();
    auto person = make_person({30, 100, 1}, {70, 100, 1}, {30, 200, 1}, {70, 200, 1});
    REQUIRE(fd.update({person}).state == FallState::Standing);
}
