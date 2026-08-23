// ModelDeploy demo_solutions —— 解决方案演示（无权重，合成轨迹）：ObjectCounter/Heatmap/SpeedEstimator/ParkingManager。
// Usage: demo_solutions
//   构建一个虚拟跨线场景，把合成 TrackResult 依次喂给各解决方案，打印计数/热力峰点/速度。
#include <cstdio>
#include <vector>

#include "vision/common/struct.h"
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/parking_manager.h"
#include "vision/tracking/base_tracker.h"

using namespace modeldeploy::vision;

static tracking::TrackResult trk(int id, float cx, float cy, int label = 0) {
    tracking::TrackResult t;
    t.track_id = id;
    t.box = Rect2f(cx - 5, cy - 5, 10, 10);
    t.label_id = label;
    t.score = 0.9f;
    return t;
}

int main() {
    solution::ObjectCounter counter;
    counter.set_line(Point2f(160, 0), Point2f(160, 240));

    solution::Heatmap heat;
    heat.set_size(320, 240);

    solution::SpeedEstimator speed;
    speed.set_meter_per_pixel(0.05f);

    solution::ParkingManager parking;
    parking.set_slots({{Point2f(0, 0), Point2f(40, 0), Point2f(40, 40), Point2f(0, 40)},
                       {Point2f(50, 0), Point2f(90, 0), Point2f(90, 40), Point2f(50, 40)}});

    // 虚拟场景：目标 1 从左到右跨线，目标 2 停在车位 1 内。
    double ts = 0.0;
    for (int f = 0; f < 30; ++f) {
        float cx1 = 100.0f + f * 5.0f;  // 100 -> 245
        std::vector<tracking::TrackResult> tracks;
        tracks.push_back(trk(1, cx1, 120));
        if (f == 0) tracks.push_back(trk(2, 20, 20));

        counter.update(tracks);
        heat.update(tracks, 320, 240);
        speed.update(tracks, ts);
        parking.update(tracks);
        ts += 33.0;
    }

    auto st = counter.stats();
    printf("object_counter: line_in=%d line_out=%d\n", st.line_in, st.line_out);
    auto peak = heat.peak();
    printf("heatmap: peak=(%d,%d)\n", peak.first, peak.second);
    auto speeds = speed.speeds_m_s();
    for (auto& kv : speeds) printf("  speed[id=%d] = %.3f m/s\n", kv.first, kv.second);
    auto occ = parking.occupancy();
    for (size_t i = 0; i < occ.size(); ++i) printf("  parking slot[%zu] occupied=%d\n", i, (int)occ[i]);
    return 0;
}
