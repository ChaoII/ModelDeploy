// ModelDeploy demo_action_skeleton：视频骨骼动作识别（ST-GCN）。
// Usage: demo_action_skeleton <stgcn.onnx> <pose.onnx> <video.mp4>
//   video::VideoDecoder（FFmpeg）抽帧 → UltralyticsPose 逐帧提关键点 → KeyPointSeq → StGcn 分类。
#include <algorithm>
#include <cstdio>
#include <vector>

#include "runtime/runtime_option.h"
#include "vision/action/keypoint_seq.h"
#include "vision/action/st_gcn.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/pose/ultralytics_pose.h"
#ifdef BUILD_VIDEO
#include "video/video_decoder.h"
#endif

namespace md = modeldeploy;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("Usage: demo_action_skeleton <stgcn.onnx> <pose.onnx> <video.mp4>\n");
        return 1;
    }
    md::RuntimeOption opt;
    opt.use_ort_backend();
    opt.set_device(modeldeploy::Device::CPU);

    md::vision::detection::UltralyticsPose pose(argv[2], opt);
    if (!pose.is_initialized()) {
        std::printf("pose init failed (missing weights?): %s\n", argv[2]);
        return 1;
    }
    md::vision::action::StGcn stgcn(argv[1], opt);
    if (!stgcn.is_initialized()) {
        std::printf("stgcn init failed (missing weights?): %s\n", argv[1]);
        return 1;
    }

#ifdef BUILD_VIDEO
    md::video::VideoDecoderConfig vcfg;
    auto dec = md::video::VideoDecoder::create(vcfg);
    if (!dec) {
        std::printf("cannot create video decoder\n");
        return 1;
    }
    std::string verr;
    if (!dec->open(argv[3], &verr)) {
        std::printf("cannot open video: %s (%s)\n", argv[3], verr.c_str());
        return 1;
    }
    md::video::VideoFrame vf;
    md::vision::action::KeyPointSeq seq;
    while (seq.frames.size() < 24 && dec->read_one_frame(&vf, &verr)) {
        std::vector<md::vision::KeyPointsResult> kps;
        if (pose.predict(vf.image, &kps) && !kps.empty()) {
            seq.frames.push_back(kps[0].keypoints);
        }
    }
    dec->close();
#else
    std::printf("BUILD_VIDEO off: demo_action_skeleton requires the video decoder\n");
    return 1;
#endif
    if (seq.frames.empty()) {
        std::printf("no skeleton frames (pose found no person in video)\n");
        return 1;
    }

    std::vector<float> scores;
    if (!stgcn.predict(seq, &scores)) {
        std::printf("predict failed\n");
        return 1;
    }
    const int best = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    std::printf("action label=%d score=%.4f\n", best, scores[best]);
    return 0;
}
