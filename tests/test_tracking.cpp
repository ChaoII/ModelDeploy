#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include "vision/tracking/base_tracker.h"
#include "vision/tracking/matching/iou_matching.h"
#include "vision/tracking/matching/hungarian.h"
#include "vision/tracking/matching/kalman_filter.h"
#include "vision/tracking/bytetrack.h"
#include "vision/tracking/botsort.h"
#include "vision/tracking/strongsort.h"
#include "vision/tracking/reid_extractor.h"
#include "vision/tracking/mot/mot_loader.h"
#include "vision/tracking/mot/clear_metrics.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tracking;
using namespace Catch;

TEST_CASE("BaseTracker: empty input yields empty output", "[tracking]") {
    auto r = empty_update();
    REQUIRE(r.empty());
}

TEST_CASE("Matching: IoU overlap", "[tracking]") {
    Rect2f a{0,0,10,10}, b{0,0,10,10};
    REQUIRE(iou(a,b) == Approx(1.0f));
}

TEST_CASE("Matching: IoU no overlap", "[tracking]") {
    Rect2f a{0,0,10,10}, b{100,100,10,10};
    REQUIRE(iou(a,b) == Approx(0.0f));
}

TEST_CASE("Matching: IoU distance", "[tracking]") {
    Rect2f a{0,0,10,10}, b{0,0,10,10};
    auto d = iou_distance({a}, {b});
    REQUIRE(d.size() == 1);
    REQUIRE(d[0].size() == 1);
    REQUIRE(d[0][0] == Approx(0.0f));
}

TEST_CASE("Matching: hungarian assignment", "[tracking]") {
    std::vector<std::vector<float>> cost{{1,2},{2,1}};
    auto res = linear_sum_assignment(cost);
    REQUIRE(res.size() == 2);
    REQUIRE(res[0].first == 0);
    REQUIRE(res[0].second == 0);
    REQUIRE(res[1].first == 1);
    REQUIRE(res[1].second == 1);
    float total = cost[res[0].first][res[0].second] + cost[res[1].first][res[1].second];
    REQUIRE(total == Approx(2.0f));
}

TEST_CASE("Matching: hungarian rectangular", "[tracking]") {
    std::vector<std::vector<float>> cost{{1,3,1},{2,1,4}};
    auto res = linear_sum_assignment(cost);
    REQUIRE(res.size() == 2);
    std::vector<int> used_cols;
    float total = 0.0f;
    for (size_t r = 0; r < res.size(); ++r) {
        REQUIRE(res[r].first == static_cast<int>(r));
        REQUIRE(std::find(used_cols.begin(), used_cols.end(), res[r].second) == used_cols.end());
        used_cols.push_back(res[r].second);
        total += cost[res[r].first][res[r].second];
    }
    REQUIRE(total == Approx(2.0f));
}

TEST_CASE("KalmanFilter: init sets nonzero position/velocity covariance diag", "[tracking]") {
    KalmanFilter kf;
    Rect2f box{10, 10, 20, 40};
    kf.init(box);
    auto diag = kf.get_covariance_diag();
    REQUIRE(diag.size() == 8);
    for (size_t i = 0; i < diag.size(); ++i) {
        INFO("state dim " << i);
        REQUIRE(diag[i] > 0.0);
    }
}

TEST_CASE("KalmanFilter: static box stays put", "[tracking]") {
    KalmanFilter kf;
    Rect2f box{10,10,20,40};
    kf.init(box);
    kf.predict();
    auto s1 = kf.get_state();
    REQUIRE(s1.x == Approx(10.0f).margin(0.1f));
    REQUIRE(s1.y == Approx(10.0f).margin(0.1f));
    REQUIRE(s1.width == Approx(20.0f).margin(0.1f));
    REQUIRE(s1.height == Approx(40.0f).margin(0.1f));
    kf.update(box);
    auto s2 = kf.get_state();
    REQUIRE(s2.x == Approx(10.0f).margin(0.1f));
    REQUIRE(s2.y == Approx(10.0f).margin(0.1f));
    REQUIRE(s2.width == Approx(20.0f).margin(0.1f));
    REQUIRE(s2.height == Approx(40.0f).margin(0.1f));
}

TEST_CASE("KalmanFilter: converges toward measurement", "[tracking]") {
    KalmanFilter kf;
    Rect2f start{0,0,20,40};
    Rect2f target{100,50,30,60};
    kf.init(start);
    Rect2f prev = target;
    for (int i = 0; i < 20; ++i) {
        kf.predict();
        kf.update(target);
        prev = kf.get_state();
    }
    REQUIRE(prev.x == Approx(100.0f).margin(2.0f));
    REQUIRE(prev.y == Approx(50.0f).margin(2.0f));
    REQUIRE(prev.width == Approx(30.0f).margin(2.0f));
    REQUIRE(prev.height == Approx(60.0f).margin(2.0f));
}

TEST_CASE("ByteTrack: stable id across frames", "[tracking]") {
    ByteTracker tr;
    Detection d1{{0,0,20,20},0.9f,0};
    auto f1 = tr.update({d1});
    Detection d2{{2,2,20,20},0.9f,0};
    auto f2 = tr.update({d2});
    REQUIRE(f1.size()==1); REQUIRE(f2.size()==1);
    REQUIRE(f1[0].track_id == f2[0].track_id);
}
TEST_CASE("ByteTrack: lost keeps id within max_age", "[tracking]") {
    ByteTracker tr;
    Detection d{{0,0,20,20},0.9f,0};
    tr.update({d});                          // 检出
    auto miss = tr.update({});               // 丢失一帧
    auto back = tr.update({d});              // 回到视野
    REQUIRE(back.size()==1);
    REQUIRE(back[0].track_id == 0);          // id 复用
}
TEST_CASE("ByteTrack: low-IoU object not merged into existing track id", "[tracking]") {
    ByteTracker tr;                          // match_thresh=0.8 -> min required IoU 0.2
    Detection a{{0,0,10,10},0.9f,0};
    auto f1 = tr.update({a});
    REQUIRE(f1.size()==1);
    REQUIRE(f1[0].track_id == 0);
    Detection b{{5,5,10,10},0.9f,0};         // IoU(a,b) ~ 0.14 -> distance ~0.86 > 0.8
    tr.update({b});                          // must NOT absorb b into id0 (gate rejects)
    auto f3 = tr.update({b});                // b is a distinct object, has a fresh id
    REQUIRE(f3.size()==1);
    REQUIRE(f3[0].track_id != 0);            // b is not the old a-track
    REQUIRE(f3[0].track_id == 1);            // b got a fresh distinct track id
}
TEST_CASE("ReidExtractor: uninitialized returns empty", "[tracking]") {
    ReidExtractor re;
    ImageData dummy;
    REQUIRE(re.extract(dummy).empty());
}

TEST_CASE("ReidExtractor: l2 normalize unit", "[tracking]") {
    ReidExtractor re;
    auto n = re.l2_normalize({3.0f, 4.0f}); // norm=5 -> {0.6,0.8}
    REQUIRE(n.size() == 2);
    REQUIRE(std::abs(n[0] - 0.6f) < 1e-5f);
    REQUIRE(std::abs(n[1] - 0.8f) < 1e-5f);
}

TEST_CASE("ReidExtractor: zero-norm l2 normalize returns unchanged", "[tracking]") {
    ReidExtractor re;
    auto n = re.l2_normalize({0.0f, 0.0f});
    REQUIRE(n.size() == 2);
    REQUIRE(n[0] == 0.0f);
    REQUIRE(n[1] == 0.0f);
    // no NaN produced
    REQUIRE(std::isfinite(n[0]));
    REQUIRE(std::isfinite(n[1]));
}

TEST_CASE("ReidExtractor: missing model file fails init", "[tracking]") {
    ReidExtractor re;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    opt.use_ort_backend();
    REQUIRE_FALSE(re.init("nonexistent_model.onnx", opt));
    REQUIRE_FALSE(re.is_initialized());
}

TEST_CASE("ByteTrack: two low-overlap objects keep distinct stable ids", "[tracking]") {
    ByteTracker tr;
    Detection a{{0,0,10,10},0.9f,0};
    Detection b{{5,5,10,10},0.9f,0};         // IoU(a,b) ~ 0.14 (partial overlap, below min)
    auto f1 = tr.update({a,b});
    REQUIRE(f1.size()==2);
    REQUIRE(f1[0].track_id != f1[1].track_id);
    auto f2 = tr.update({a,b});
    REQUIRE(f2.size()==2);
    REQUIRE(f2[0].track_id != f2[1].track_id);
}

TEST_CASE("BoT-SORT: no frames still stable ids", "[tracking]") {
    BotSortTracker tr;
    Detection d1{{0,0,20,20},0.9f,0};
    auto f1 = tr.update({d1});
    Detection d2{{2,2,20,20},0.9f,0};
    auto f2 = tr.update({d2});
    REQUIRE(f1.size()==1);
    REQUIRE(f2.size()==1);
    REQUIRE(f1[0].track_id == f2[0].track_id);
}
TEST_CASE("BoT-SORT: distinct appearance keeps distinct ids at same spot", "[tracking]") {
    BotSortTracker tr;
    Detection a{{0,0,20,20},0.9f,0,{1.0f,0.0f,0.0f}};
    Detection b{{0,0,20,20},0.9f,0,{0.0f,1.0f,0.0f}};
    tr.update({a,b});
    auto f2 = tr.update({a,b});
    REQUIRE(f2.size()==2);
    REQUIRE(f2[0].track_id != f2[1].track_id);
}
TEST_CASE("BoT-SORT: EMA feature tracking keeps id amid appearance drift", "[tracking]") {
    BotSortTracker tr;
    std::vector<float> f0{1,0,0}, f1{0.9f,0.2f,0}, f2{0.8f,0.4f,0};
    Detection d0{{0,0,20,20},0.9f,0,f0};
    tr.update({d0});
    auto r1 = tr.update({Detection{{2,2,20,20},0.9f,0,f1}});
    auto r2 = tr.update({Detection{{4,4,20,20},0.9f,0,f2}});
    REQUIRE(r2.size()==1);
}

// This test is deliberately DISCRIMINATING: a pure-IoU implementation FAILS it,
// but the fused IoU+appearance cost passes it. Frame 1 seeds one track (id0)
// with feature {1,0,0}. Frame 2 presents two detections near the SAME spot: the
// true object B (feature {1,0,0}, slightly offset box) and a foreign object C
// (feature {0,1,0}) whose box sits EXACTLY on the track's predicted position, so
// C has the *better* IoU with id0. Pure IoU would thereby match the foreign C to
// id0 (wrong); the fused cost uses the ~0 cosine distance to keep B on id0.
TEST_CASE("BoT-SORT: appearance prevents foreign feature from stealing track id", "[tracking]") {
    BotSortTracker tr; // frame==nullptr -> CMC is a no-op (identity)
    Detection a{{100,100,20,20},0.9f,0,{1.0f,0.0f,0.0f}};
    auto f1 = tr.update({a}); // seeds id0 with feature {1,0,0}
    REQUIRE(f1.size() == 1);
    REQUIRE(f1[0].track_id == 0);

    // True object B: same appearance, slight box offset (IoU(track,B) < 1).
    Detection b{{102,102,20,20},0.9f,0,{1.0f,0.0f,0.0f}};
    // Foreign object C: different appearance, box exactly on predicted position
    // (IoU(track,C) == 1 > IoU(track,B)) so pure IoU would wrongly pick C.
    Detection c{{100,100,20,20},0.9f,0,{0.0f,1.0f,0.0f}};
    auto f2 = tr.update({b, c});
    REQUIRE(f2.size() == 2);                      // two distinct objects
    REQUIRE(f2[0].track_id != f2[1].track_id);

    // The true-feature {1,0,0} object must still own id0 (appearance beat IoU).
    bool true_on_id0 = false;
    bool foreign_seen = false;
    for (const auto& r : f2) {
        if (r.feature.size() >= 2 && r.feature[0] > 0.9f) {   // ~{1,0,0}
            REQUIRE(r.track_id == 0);
            true_on_id0 = true;
        }
        if (r.feature.size() >= 2 && r.feature[1] > 0.9f) {   // ~{0,1,0}
            foreign_seen = true;
        }
    }
    REQUIRE(true_on_id0);
    REQUIRE(foreign_seen);

    // Re-feed the pair on frame 3: id0 stays glued to the true appearance.
    auto f3 = tr.update({b, c});
    REQUIRE(f3.size() == 2);
    for (const auto& r : f3) {
        if (r.feature.size() >= 2 && r.feature[0] > 0.9f) {
            REQUIRE(r.track_id == 0);
        }
    }
}

// DISCRIMINATING against "copy latest det feature" implementations: asserts the
// EMA actually blends. Frame 1 seeds id0's EMA = A = {1,0,0}. Frame 2 matches a
// single det B = {0,1,0}. ema_alpha = 0.9 => blended pre-normalize v = 0.9*A +
// 0.1*B = {0.9, 0.1, 0}, then renormalized. A pure "copy latest" would give B;
// "no averaging" would give A; this asserts the actual average.
TEST_CASE("BoT-SORT: EMA feature is blended, not copied", "[tracking]") {
    BotSortTracker tr;
    std::vector<float> A{1.0f, 0.0f, 0.0f};
    std::vector<float> B{0.0f, 1.0f, 0.0f};
    auto f1 = tr.update({Detection{{0,0,20,20},0.9f,0,A}});
    REQUIRE(f1.size() == 1);
    REQUIRE(f1[0].track_id == 0);
    auto f2 = tr.update({Detection{{2,2,20,20},0.9f,0,B}});
    REQUIRE(f2.size() == 1);
    REQUIRE(f2[0].track_id == 0);

    const float alpha = 0.9f; // default ema_alpha_
    const float e0 = alpha * 1.0f + (1.0f - alpha) * 0.0f; // 0.9
    const float e1 = alpha * 0.0f + (1.0f - alpha) * 1.0f; // 0.1
    const float e2 = 0.0f;
    const float norm = std::sqrt(e0 * e0 + e1 * e1 + e2 * e2);

    REQUIRE(f2[0].feature.size() == 3);
    REQUIRE(f2[0].feature[0] == Approx(e0 / norm).margin(1e-3f));
    REQUIRE(f2[0].feature[1] == Approx(e1 / norm).margin(1e-3f));
    REQUIRE(f2[0].feature[2] == Approx(0.0f).margin(1e-3f));
}

TEST_CASE("StrongSORT: no frames stable ids", "[tracking]") {
    StrongSortTracker tr;
    Detection d1{{0,0,20,20},0.9f,0};
    auto f1 = tr.update({d1});
    Detection d2{{2,2,20,20},0.9f,0};
    auto f2 = tr.update({d2});
    REQUIRE(f1.size()==1); REQUIRE(f2.size()==1);
    REQUIRE(f1[0].track_id == f2[0].track_id);
}

// DISCRIMINATING: a pure-IoU implementation FAILS this; appearance-priority
// (StrongSORT, appearance_priority=0.7) passes it. Frame 1 seeds id0 with
// feature {1,0,0}. Frame 2 presents two detections near the SAME spot: the true
// object B (feature {1,0,0}, slightly offset box) and a foreign object C
// (feature {0,1,0}) whose box sits EXACTLY on the track's predicted position, so
// C has the better IoU with id0. Pure IoU would match the foreign C to id0
// (cost_C = 1-IoU = 0 vs cost_B = 0.32); appearance-priority routes B to id0
// (cost_B = 0.7*0 + 0.3*0.32 = 0.096 vs cost_C = 0.7*1.0 + 0.3*0 = 0.7).
TEST_CASE("StrongSORT: appearance prevents foreign re-association", "[tracking]") {
    StrongSortTracker tr;
    Detection a{{100,100,20,20},0.9f,0,{1.0f,0.0f,0.0f}};
    auto f1 = tr.update({a});   // seeds id0 with feature {1,0,0}
    REQUIRE(f1.size() == 1);
    REQUIRE(f1[0].track_id == 0);

    // True object B: same appearance, slight box offset (IoU(track,B) < 1).
    Detection b{{102,102,20,20},0.9f,0,{1.0f,0.0f,0.0f}};
    // Foreign object C: different appearance, box exactly on predicted position
    // (IoU(track,C) == 1 > IoU(track,B)) so pure IoU would wrongly pick C.
    Detection c{{100,100,20,20},0.9f,0,{0.0f,1.0f,0.0f}};
    auto f2 = tr.update({b, c});
    REQUIRE(f2.size() == 2);                      // two distinct objects
    REQUIRE(f2[0].track_id != f2[1].track_id);

    // The true-feature {1,0,0} object must own id0 (appearance beat IoU).
    bool true_on_id0 = false;
    bool foreign_seen = false;
    for (const auto& r : f2) {
        if (r.feature.size() >= 2 && r.feature[0] > 0.9f) {   // ~{1,0,0}
            REQUIRE(r.track_id == 0);
            true_on_id0 = true;
        }
        if (r.feature.size() >= 2 && r.feature[1] > 0.9f) {   // ~{0,1,0}
            foreign_seen = true;
        }
    }
    REQUIRE(true_on_id0);
    REQUIRE(foreign_seen);

    // Re-feed the pair on frame 3: id0 stays glued to the true appearance.
    auto f3 = tr.update({b, c});
    REQUIRE(f3.size() == 2);
    for (const auto& r : f3) {
        if (r.feature.size() >= 2 && r.feature[0] > 0.9f) {
            REQUIRE(r.track_id == 0);
        }
    }
}

// DISCRIMINATING against "copy latest det feature" implementations: asserts the
// EMA actually blends. Frame 1 seeds id0's EMA = A = {1,0,0}. Frame 2 matches a
// single det B = {0,1,0}. ema_alpha = 0.9 => blended pre-normalize v = 0.9*A +
// 0.1*B = {0.9, 0.1, 0}, renormalized. A "copy latest" would give B; "no
// averaging" would give A; this asserts the actual average.
TEST_CASE("StrongSORT: EMA blend asserted", "[tracking]") {
    StrongSortTracker tr;
    std::vector<float> A{1.0f, 0.0f, 0.0f};
    std::vector<float> B{0.0f, 1.0f, 0.0f};
    auto f1 = tr.update({Detection{{0,0,20,20},0.9f,0,A}});
    REQUIRE(f1.size() == 1);
    REQUIRE(f1[0].track_id == 0);
    auto r = tr.update({Detection{{2,2,20,20},0.9f,0,B}});
    REQUIRE(r.size() == 1);
    REQUIRE(r[0].track_id == 0);
    // alpha=0.9: 0.9*{1,0,0}+0.1*{0,1,0}={0.9,0.1,0}, norm=sqrt(0.82)~0.90554
    // normalized => {0.99388, 0.11043, 0}
    REQUIRE(r[0].feature.size() == 3);
    REQUIRE(std::abs(r[0].feature[0] - 0.99388f) < 1e-4);
    REQUIRE(std::abs(r[0].feature[1] - 0.11043f) < 1e-4);
    REQUIRE(std::abs(r[0].feature[2]) < 1e-4);
}

TEST_CASE("StrongSORT: lost keeps id within max_age", "[tracking]") {
    StrongSortTracker tr;
    Detection d{{0,0,20,20},0.9f,0};
    tr.update({d});             // 检出
    auto miss = tr.update({});  // 丢失一帧
    auto back = tr.update({d}); // 回到视野
    REQUIRE(back.size() == 1);
    REQUIRE(back[0].track_id == 0); // id 复用
}

// RAII temp file helper for MOT loader tests (never committed to repo).
namespace {
struct TempMotFile {
    std::filesystem::path path;
    explicit TempMotFile(const std::string& content) {
        path = std::filesystem::temp_directory_path() /
               ("mot_test_" + std::to_string(std::rand()) + std::to_string(std::rand()) + ".txt");
        std::ofstream os(path, std::ios::trunc);
        os << content;
    }
    ~TempMotFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};
}  // namespace

TEST_CASE("MotLoader: empty / nonexistent file yields empty sequence", "[tracking]") {
    auto seq = load_mot_sequence("nonexistent_mot_file_xyz.txt");
    REQUIRE(seq.frames.empty());
    std::vector<std::vector<TrackResult>> preds;   // empty seq -> empty predictions
    auto m = compute_clear(seq, preds);
    REQUIRE(m.mota == Approx(1.0f));               // vacuous perfect (num_gt == 0)
    REQUIRE(m.idf1 == Approx(0.0f));
    REQUIRE(m.num_gt == 0);
}

TEST_CASE("MotLoader: parses MOTChallenge text into GT + dets, perfect prediction -> MOTA==1", "[tracking]") {
    std::string mot =
        "1,1,0,0,10,10,-1,-1,-1\n"
        "1,2,100,100,10,10,-1,-1,-1\n"
        "1,-1,0,0,10,10,0.9,1,-1\n"
        "1,-1,100,100,10,10,0.8,1,-1\n"
        "2,1,0,0,10,10,-1,-1,-1\n"
        "2,2,100,100,10,10,-1,-1,-1\n"
        "2,-1,0,0,10,10,0.85,1,-1\n"
        "2,-1,100,100,10,10,0.75,1,-1\n";
    TempMotFile f(mot);

    auto seq = load_mot_sequence(f.path.string());
    REQUIRE(seq.frames.size() == 2);
    REQUIRE(seq.frames[0].frame_id == 1);
    REQUIRE(seq.frames[1].frame_id == 2);
    REQUIRE(seq.num_gt_ids == 2);
    REQUIRE(seq.frames[0].gt_boxes.size() == 2);
    REQUIRE(seq.frames[0].gt_ids[0] == 1);
    REQUIRE(seq.frames[0].gt_ids[1] == 2);
    REQUIRE(seq.frames[0].dets.size() == 2);
    REQUIRE(seq.frames[0].dets[0].score == Approx(0.9f));
    REQUIRE(seq.frames[0].dets[1].score == Approx(0.8f));
    REQUIRE(seq.frames[1].dets[0].score == Approx(0.85f));

    // PERFECT predictions: each GT box with GT id as track id.
    std::vector<std::vector<TrackResult>> perfect(seq.frames.size());
    for (size_t i = 0; i < seq.frames.size(); ++i) {
        const auto& fr = seq.frames[i];
        for (size_t j = 0; j < fr.gt_boxes.size(); ++j) {
            TrackResult tr;
            tr.track_id = fr.gt_ids[j];
            tr.box = fr.gt_boxes[j];
            tr.score = 1.0f;
            tr.label_id = 0;
            perfect[i].push_back(tr);
        }
    }
    auto m = compute_clear(seq, perfect);
    REQUIRE(m.mota == Approx(1.0f));
    REQUIRE(m.idf1 >= Approx(0.99f));
    REQUIRE(m.num_fp == 0);
    REQUIRE(m.num_fn == 0);
    REQUIRE(m.precision == Approx(1.0f));
    REQUIRE(m.recall == Approx(1.0f));

    // Predictions that MISS every GT -> MOTA == 0, all FN.
    std::vector<std::vector<TrackResult>> miss(seq.frames.size());
    auto mm = compute_clear(seq, miss);
    REQUIRE(mm.mota == Approx(0.0f));
    REQUIRE(mm.num_fn == 4);               // 2 frames x 2 GT
    REQUIRE(mm.recall == Approx(0.0f));
}

TEST_CASE("MotLoader: detects ID switch when a track swaps GT identity", "[tracking]") {
    MotSequence seq;
    MotFrame f1, f2;
    f1.frame_id = 1;
    f1.gt_boxes.push_back({0, 0, 10, 10});
    f1.gt_ids.push_back(1);
    f2.frame_id = 2;
    f2.gt_boxes.push_back({100, 100, 10, 10});
    f2.gt_ids.push_back(2);
    seq.frames = {f1, f2};

    std::vector<std::vector<TrackResult>> preds(2);
    TrackResult a; a.track_id = 0; a.box = {0, 0, 10, 10};
    TrackResult b; b.track_id = 0; b.box = {100, 100, 10, 10};
    preds[0].push_back(a);   // track 0 -> GT 1
    preds[1].push_back(b);   // track 0 -> GT 2  (identity swapped)

    auto m = compute_clear(seq, preds);
    REQUIRE(m.id_switches >= 1);
}
