// GPU(及 Jetson 同类 CUDA)设备侧 vis_* 渲染存图工具：真实图像 → NV12 → 逐方法绘制 → 回读 PNG。
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"
#include "vision/processors/cuda/cuda_processor_backend.h"

using namespace modeldeploy::vision;

struct Frame {
    int w = 0, h = 0;
    std::vector<uint8_t> y, uv;
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    std::vector<uint8_t> back_y, back_uv;
    bool upload() {
        return cudaMemcpy(d_y, y.data(), y.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(d_uv, uv.data(), uv.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool readback() {
        back_y.resize(y.size()); back_uv.resize(uv.size());
        return cudaMemcpy(back_y.data(), d_y, y.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
               cudaMemcpy(back_uv.data(), d_uv, uv.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    ImageData make() {
        const ImageData::Plane pl[2] = {{d_y, w}, {d_uv, w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::GPU);
    }
};

static void bgr_to_nv12(const cv::Mat& bgr, std::vector<uint8_t>& y, std::vector<uint8_t>& uv) {
    const int w = bgr.cols, h = bgr.rows;
    y.assign(static_cast<size_t>(w) * h, 0);
    uv.assign(static_cast<size_t>(w) * (h / 2), 0);
    for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx) {
            const cv::Vec3b& p = bgr.at<cv::Vec3b>(yy, xx);
            const int B = p[0], G = p[1], R = p[2];
            y[static_cast<size_t>(yy) * w + xx] = static_cast<uint8_t>((66 * R + 129 * G + 25 * B + 128) / 256 + 16);
            if ((yy & 1) == 0 && (xx & 1) == 0) {
                const int i = (yy / 2) * w + xx;
                uv[i] = static_cast<uint8_t>((-38 * R - 74 * G + 112 * B + 128) / 256 + 128);
                uv[i + 1] = static_cast<uint8_t>((112 * R - 94 * G - 18 * B + 128) / 256 + 128);
            }
        }
}

static cv::Mat nv12_to_bgr(const std::vector<uint8_t>& y, const std::vector<uint8_t>& uv, int w, int h) {
    cv::Mat bgr(h, w, CV_8UC3);
    auto clamp8 = [](int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx) {
            const int C = static_cast<int>(y[static_cast<size_t>(yy) * w + xx]) - 16;
            const int ui = (yy / 2) * w + (xx & ~1);
            const int D = static_cast<int>(uv[ui]) - 128;
            const int E = static_cast<int>(uv[ui + 1]) - 128;
            cv::Vec3b& o = bgr.at<cv::Vec3b>(yy, xx);
            o[2] = clamp8((298 * C + 409 * E + 128) >> 8);
            o[1] = clamp8((298 * C - 100 * D - 208 * E + 128) >> 8);
            o[0] = clamp8((298 * C + 516 * D + 128) >> 8);
        }
    return bgr;
}

static int do_none(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    (void)fr; (void)b; (void)o; return 0;
}
static int do_det(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {    DetectionResult r; r.box = {fr.w * 0.10f, fr.h * 0.15f, fr.w * 0.45f, fr.h * 0.50f}; r.label_id = 1; r.score = 0.92f;
    DetectionResult r2; r2.box = {fr.w * 0.58f, fr.h * 0.20f, fr.w * 0.36f, fr.h * 0.45f}; r2.label_id = 3; r2.score = 0.78f;
    std::vector<DetectionResult> v{r, r2};
    return b.vis_det_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_obb(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    ObbResult r; r.rotated_box = RotatedRect(fr.w * 0.45f, fr.h * 0.55f, fr.w * 0.34f, fr.h * 0.18f, 28.f);
    r.label_id = 2; r.score = 0.83f;
    std::vector<ObbResult> v{r};
    return b.vis_obb_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_pose(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    KeyPointsResult r; r.box = {fr.w * 0.20f, fr.h * 0.15f, fr.w * 0.35f, fr.h * 0.70f};
    r.label_id = 2; r.score = 0.9f;
    const float cx = fr.w * 0.38f, cy = fr.h * 0.35f, sx = fr.w * 0.30f, sy = fr.h * 0.30f;
    r.keypoints = { {cx, cy - sy * 2.0f, 1.f}, {cx, cy - sy * 1.5f, 1.f}, {cx, cy - sy, 1.f},
                    {cx - sx * 0.8f, cy - sy * 0.5f, 1.f}, {cx - sx, cy, 1.f},
                    {cx + sx * 0.8f, cy - sy * 0.5f, 1.f}, {cx + sx, cy, 1.f},
                    {cx - sx, cy + sy * 0.6f, 1.f}, {cx - sx * 1.2f, cy + sy * 1.5f, 1.f},
                    {cx + sx, cy + sy * 0.6f, 1.f}, {cx + sx * 1.2f, cy + sy * 1.5f, 1.f},
                    {cx, cy + sy * 0.6f, 1.f}, {cx - sx * 0.4f, cy + sy * 1.2f, 1.f},
                    {cx + sx * 0.4f, cy + sy * 1.2f, 1.f}, {cx, cy + sy * 1.6f, 1.f},
                    {cx - sx * 0.4f, cy + sy * 1.8f, 1.f}, {cx + sx * 0.4f, cy + sy * 1.8f, 1.f} };
    std::vector<KeyPointsResult> v{r};
    return b.vis_pose_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_kpt(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    KeyPointsResult r; r.box = {fr.w * 0.1f, fr.h * 0.1f, fr.w * 0.5f, fr.h * 0.5f}; r.label_id = 0; r.score = 0.85f;
    for (int i = 0; i < 17; ++i) r.keypoints.emplace_back(fr.w * (0.2f + 0.03f * i), fr.h * (0.3f + 0.02f * i), 1.f);
    std::vector<KeyPointsResult> v{r};
    return b.vis_keypoints_nv12(fr.make(), v, o, false) ? 0 : 1;
}
static int do_hand(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    KeyPointsResult r; r.box = {fr.w * 0.3f, fr.h * 0.3f, fr.w * 0.3f, fr.h * 0.3f}; r.label_id = 0; r.score = 0.88f;
    const float cx = fr.w * 0.45f, cy = fr.h * 0.45f, s = fr.w * 0.1f;
    r.keypoints = { {cx, cy, 1.f}, {cx + s, cy - s, 1.f}, {cx + s * 1.4f, cy - s * 0.6f, 1.f}, {cx + s * 1.5f, cy - s * 0.2f, 1.f}, {cx + s * 1.6f, cy + s * 0.2f, 1.f},
                    {cx - s, cy - s, 1.f}, {cx - s * 1.3f, cy - s * 0.4f, 1.f}, {cx - s * 1.3f, cy, 1.f}, {cx - s * 1.2f, cy + s * 0.4f, 1.f},
                    {cx, cy - s * 1.5f, 1.f}, {cx + s * 0.4f, cy - s * 1.2f, 1.f}, {cx + s * 0.5f, cy - s * 0.9f, 1.f}, {cx + s * 0.4f, cy - s * 0.6f, 1.f},
                    {cx + s, cy - s * 1.3f, 1.f}, {cx + s * 1.1f, cy - s, 1.f}, {cx + s * 1.1f, cy - s * 0.7f, 1.f}, {cx + s, cy - s * 0.4f, 1.f},
                    {cx + s * 1.5f, cy - s * 1.1f, 1.f}, {cx + s * 1.6f, cy - s * 0.9f, 1.f}, {cx + s * 1.6f, cy - s * 0.6f, 1.f}, {cx + s * 1.5f, cy - s * 0.3f, 1.f} };
    std::vector<KeyPointsResult> v{r};
    return b.vis_hand_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_ocr(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    OCRResult r;
    auto box = [&](float x, float y, float w, float h) {
        return std::array<int, 8>{ (int)(x*fr.w), (int)(y*fr.h), (int)((x+w)*fr.w), (int)(y*fr.h),
                                   (int)((x+w)*fr.w), (int)((y+h)*fr.h), (int)(x*fr.w), (int)((y+h)*fr.h) };
    };
    r.boxes.push_back(box(0.1f, 0.1f, 0.4f, 0.05f)); r.text.push_back("Hello World 2024");
    r.boxes.push_back(box(0.1f, 0.2f, 0.5f, 0.05f)); r.text.push_back("Line two text");
    return b.vis_ocr_nv12(fr.make(), r, o) ? 0 : 1;
}
static int do_lpr(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    LprResult r; r.box = {fr.w * 0.25f, fr.h * 0.5f, fr.w * 0.5f, fr.h * 0.12f};
    r.score = 0.9f; r.car_plate_str = "JING A12345"; r.car_plate_color = "blue";
    std::vector<LprResult> v{r};
    return b.vis_lpr_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_attr(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    AttributeResult r; r.box = {fr.w * 0.3f, fr.h * 0.2f, fr.w * 0.3f, fr.h * 0.55f};
    r.box_score = 0.85f; r.attr_scores = {0.9f, 0.1f, 0.8f, 0.7f};
    std::vector<AttributeResult> v{r};
    std::vector<int> ab{0};
    return b.vis_attr_nv12(fr.make(), v, o, ab, true) ? 0 : 1;
}
static int do_cls(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    ClassifyResult r; r.label_ids = {1, 3}; r.scores = {0.85f, 0.60f};
    return b.vis_cls_nv12(fr.make(), r, o, 2) ? 0 : 1;
}
static int do_iseg(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    InstanceSegResult r; r.box = {fr.w * 0.15f, fr.h * 0.2f, fr.w * 0.45f, fr.h * 0.55f};
    r.label_id = 0; r.score = 0.9f; r.mask.shape = {fr.h / 2, fr.w / 2};
    r.mask.buffer.assign(static_cast<size_t>(fr.w / 2) * (fr.h / 2), 1);
    std::vector<InstanceSegResult> v{r};
    return b.vis_iseg_nv12(fr.make(), v, o) ? 0 : 1;
}
static int do_sem(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    const int gw = fr.w / 32, gh = fr.h / 32;
    SemSegResult r; r.shape = {gh, gw}; r.num_classes = 20;
    r.labels.assign(static_cast<size_t>(gh) * gw, 0);
    for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x)
            r.labels[static_cast<size_t>(y) * gw + x] = static_cast<uint8_t>((y * 3 + x) % 20);
    return b.vis_sem_nv12(fr.make(), r, o) ? 0 : 1;
}
static int do_depth(Frame& fr, CudaProcessorBackend& b, VisionProcessorBackend::VisOptions& o) {
    DepthResult r; r.shape = {fr.h, fr.w};
    r.depth.assign(static_cast<size_t>(fr.h) * fr.w, 0.f);
    for (int y = 0; y < fr.h; ++y)
        for (int x = 0; x < fr.w; ++x)
            r.depth[static_cast<size_t>(y) * fr.w + x] = static_cast<float>(x + y) / static_cast<float>(fr.w + fr.h);
    return b.vis_depth_nv12(fr.make(), r, o, true) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: render_vis_gpu <input.jpg> <out_prefix> [font.ttf/.ttc] [reps]\n"); return 2; }
    const cv::Mat rgb = cv::imread(argv[1]);
    if (rgb.empty()) { printf("FAIL read image\n"); return 2; }
    const int w = rgb.cols, h = rgb.rows;
    printf("image %dx%d\n", w, h);
    Frame fr; fr.w = w; fr.h = h;
    bgr_to_nv12(rgb, fr.y, fr.uv);
    if (cudaMalloc(&fr.d_y, fr.y.size()) != cudaSuccess || cudaMalloc(&fr.d_uv, fr.uv.size()) != cudaSuccess) {
        printf("FAIL cudaMalloc\n"); return 2;
    }
    CudaProcessorBackend backend;
    VisionProcessorBackend::VisOptions opt;
    opt.threshold = 0.3f; opt.alpha = 0.35f;
    if (argc >= 4) opt.font_path = argv[3];
    std::string pre = argv[2];
    const int reps = argc >= 5 ? std::atoi(argv[4]) : 30;
    bool allok = true;
    auto run = [&](const char* tag, int (*doit)(Frame&, CudaProcessorBackend&, VisionProcessorBackend::VisOptions&)) {
        // 预热：跑一遍丢弃（加载字体、驱动初始化等一次性成本不计入稳态）
        fr.upload();
        if (doit(fr, backend, opt) == 0) cudaDeviceSynchronize();
        // 计时：多次执行，取 best
        double best = 1e18, sum = 0;
        for (int r = 0; r < reps; ++r) {
            fr.upload();
            const auto t0 = std::chrono::high_resolution_clock::now();
            int rc = doit(fr, backend, opt);
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
            sum += ms;
        }
        fr.readback();
        std::string out = pre + std::string("_") + tag + std::string(".png");
        cv::Mat bgr = nv12_to_bgr(fr.back_y, fr.back_uv, fr.w, fr.h);
        bool saved = cv::imwrite(out, bgr);
        printf("[%s] rc=%d draw_best=%.3fms avg=%.3fms (reps=%d) saved=%d\n",
               tag, saved ? 0 : 1, best, sum / reps, reps, saved ? 1 : 0);
        if (!saved) allok = false;
    };
    run("det", do_det);
    run("base", do_none);
    run("obb", do_obb);
    run("pose", do_pose);
    run("keypoints", do_kpt);
    run("hand", do_hand);
    run("ocr", do_ocr);
    run("lpr", do_lpr);
    run("attr", do_attr);
    run("cls", do_cls);
    run("iseg", do_iseg);
    run("sem", do_sem);
    run("depth", do_depth);
    cudaFree(fr.d_y); cudaFree(fr.d_uv);
    printf(allok ? "ALL_RENDER_OK\n" : "SOME_RENDER_FAILED\n");
    return allok ? 0 : 1;
}
