// ModelDeploy demo_action：视频动作识别（TSN，RGB 帧）。
// Usage: demo_action <tsn.onnx> <video.mp4>
//   BUILD_VIDEO：用 video::VideoDecoder（FFmpeg）抽帧 → TSN.predict → 打印 top3 动作 label+score。
//   非 BUILD_VIDEO（退化）：参数改为若干帧图片路径。
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "runtime/runtime_option.h"
#include "vision/action/tsn.h"
#include "vision/common/image_data.h"
#ifdef BUILD_VIDEO
#include "video/video_decoder.h"
#endif

namespace md = modeldeploy;

static void print_topk(const std::vector<float>& scores, int k) {
    std::vector<int> idx(scores.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    const size_t n = std::min<size_t>(static_cast<size_t>(k), idx.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    for (size_t i = 0; i < n; ++i) {
        std::printf("top%zu label=%d score=%.4f\n", i + 1, idx[i], scores[idx[i]]);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage: demo_action <tsn.onnx> <video.mp4>\n");
        std::printf("  Uses video::VideoDecoder (FFmpeg) to sample frames, then TSN predicts top-K actions.\n");
        return 1;
    }
    md::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();

    md::vision::action::TSN model(argv[1], opt);
    if (!model.is_initialized()) {
        std::printf("init failed (missing weights?): %s\n", argv[1]);
        return 1;
    }

#ifdef BUILD_VIDEO
    md::video::VideoDecoder dec;
    if (!dec.open(argv[2])) {
        std::printf("cannot open video: %s\n", argv[2]);
        return 1;
    }
    std::vector<md::vision::ImageData> frames;
    md::vision::ImageData f;
    uint64_t pts = 0;
    while (frames.size() < 32 && dec.next(&f, &pts)) {
        // VideoDecoder 抽帧为 CPU NV12 → 转 PKG_BGR（TSN 预处理内部再转 RGB）
        md::vision::ImageData bgr =
            md::vision::ImageData::cvt_color(f, ColorConvertType::CVT_NV122PKG_BGR);
        if (!bgr.empty()) frames.push_back(bgr);
    }
    dec.close();
    if (frames.empty()) {
        std::printf("no frames decoded\n");
        return 1;
    }
    std::vector<float> scores;
    if (!model.predict(frames, &scores)) {
        std::printf("predict failed\n");
        return 1;
    }
    print_topk(scores, 3);
#else
    std::vector<md::vision::ImageData> frames;
    for (int i = 2; i < argc; ++i) {
        md::vision::ImageData im = md::vision::ImageData::imread(argv[i]);
        if (!im.empty()) frames.push_back(im);
    }
    if (frames.empty()) {
        std::printf("no frames from image args (BUILD_VIDEO off)\n");
        return 1;
    }
    std::vector<float> scores;
    if (!model.predict(frames, &scores)) {
        std::printf("predict failed\n");
        return 1;
    }
    print_topk(scores, 3);
#endif
    return 0;
}
