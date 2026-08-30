// ModelDeploy demo_solutions —— 解决方案演示（无权重，合成轨迹）：ObjectCounter/RegionCounter/QueueManager/TrackZone/Heatmap/SpeedEstimator/ParkingManager。
// Usage: demo_solutions
//   构建一个虚拟跨线场景，把合成 TrackResult 依次喂给各解决方案，打印计数/热力峰点/速度。
#include <cstdio>
#include <vector>

#include "vision/common/struct.h"
#include "vision/solutions/object_counter.h"
#include "vision/solutions/region_counter.h"
#include "vision/solutions/queue_manager.h"
#include "vision/solutions/track_zone.h"
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

    // 区域场景：一个方形区域 (100,0)-(160,240)，多目标逐帧进出，演示三方案。
    std::vector<Point2f> region = {Point2f(100, 0), Point2f(160, 0), Point2f(160, 240), Point2f(100, 240)};

    solution::RegionCounter rc;
    rc.add_region("roi", region);

    solution::QueueManager queue;
    queue.set_region(region);

    solution::TrackZone zone;
    zone.set_region(region);

    // 目标 10 在区域内停留，目标 11 从左到右穿越，目标 12 一直在区域外。
    for (int f = 0; f < 10; ++f) {
        float cx11 = 80.0f + f * 12.0f;  // 80 -> 188，途中穿过 [100,160]
        std::vector<tracking::TrackResult> tracks;
        tracks.push_back(trk(10, 130, 120));  // 始终在区域内
        tracks.push_back(trk(11, cx11, 60));  // 逐帧穿越
        tracks.push_back(trk(12, 20, 200));   // 始终在区域外
        rc.update(tracks);
        queue.update(tracks);
        zone.update(tracks);
        if (f == 4) {
            auto cnt = rc.region_counts();
            printf("region_counter[roi] = %d\n", cnt["roi"]);
            printf("queue_manager: queue_count=%d\n", queue.queue_count());
            printf("track_zone: inside_count=%d\n", zone.inside_count());
            for (const auto& t : zone.inside_tracks())
                printf("  inside id=%d label=%d\n", t.track_id, t.label_id);
        }
    }
    return 0;
}
