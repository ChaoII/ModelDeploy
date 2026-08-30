// ModelDeploy demo: 跌倒检测（姿态规则版 FallDetector）。
// 用 UltralyticsPose 对输入帧做姿态估计，把关键点逐帧喂给 FallDetector，
// 输出当前跌倒状态（Standing/PreFall/Fallen）与置信度。
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/vision/solutions/fall_detector.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main() {
    const char* model = "../../test_data/test_models/onnx/yolo11n/yolo11n-pose.onnx";
    const char* image = "../../test_data/test_images/test_detection0.jpg";

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    auto pose = std::make_unique<modeldeploy::vision::detection::UltralyticsPose>(model, opt);
    if (!pose->is_initialized()) {
        std::fprintf(stderr, "pose model init failed: %s\n", model);
        return 1;
    }
    auto img = modeldeploy::vision::ImageData::imread(image);
    if (img.empty()) {
        std::fprintf(stderr, "cannot read image: %s\n", image);
        return 1;
    }

    modeldeploy::vision::solution::FallDetector fall;

    // 逐"帧"喂入同一帧姿态（示意：真实场景应逐视频帧喂入）
    const int frames = 5;
    for (int f = 0; f < frames; ++f) {
        std::vector<modeldeploy::vision::KeyPointsResult> persons;
        if (!pose->predict(img, &persons)) {
            std::fprintf(stderr, "pose predict failed\n");
            return 1;
        }
        auto r = fall.update(persons);
        const char* names[] = {"Standing", "PreFall", "Fallen"};
        std::printf("frame %d: state=%s confidence=%.3f persons=%zu\n",
                    f, names[static_cast<int>(r.state)], r.confidence, persons.size());
    }
    return 0;
}
