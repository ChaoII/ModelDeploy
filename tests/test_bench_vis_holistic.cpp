// 整体(全场景,含数据传输)可视化绘制性能基准：对"硬件解码出的设备帧"对比
//   ① 设备端绘制(CUDA vis_*,零拷贝) ② D2H 拷贝(整帧 NV12) ③ 主机绘制(CPU 图元)  与 ④ host总耗时 = D2H + CPU。
// 场景:det / pose / ocr / cls(标量为主)。Sophgo 端用同一份场景数据移植。
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
#include "vision/processors/cpu/cpu_processor_backend.h"

using namespace modeldeploy::vision;

struct Frame {
    int w = 0, h = 0;
    std::vector<uint8_t> y, uv, by, buv;
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    bool upload() { return cudaMemcpy(d_y, y.data(), y.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
                           cudaMemcpy(d_uv, uv.data(), uv.size(), cudaMemcpyHostToDevice) == cudaSuccess; }
    double d2h_ms() {  // 整帧 NV12 D2H 拷贝
        const auto t0 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(by.data(), d_y, y.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(buv.data(), d_uv, uv.size(), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    ImageData dev() {
        const ImageData::Plane pl[2] = {{d_y, w}, {d_uv, w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::GPU);
    }
    ImageData host_img(const std::vector<uint8_t>& yy, const std::vector<uint8_t>& uu) {
        const ImageData::Plane pl[2] = {{const_cast<uint8_t*>(yy.data()), w}, {const_cast<uint8_t*>(uu.data()), w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU);
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

// ── 设备端(CUDA)高层 vis_* 绘制 ──
static bool run_dev(Frame& fr, CudaProcessorBackend& b, int scene,
                    VisionProcessorBackend::VisOptions& opt) {
    switch (scene) {
        case 0: {
            DetectionResult r; r.box = {fr.w * 0.10f, fr.h * 0.15f, fr.w * 0.45f, fr.h * 0.50f}; r.label_id = 1; r.score = 0.92f;
            DetectionResult r2; r2.box = {fr.w * 0.58f, fr.h * 0.20f, fr.w * 0.36f, fr.h * 0.45f}; r2.label_id = 3; r2.score = 0.78f;
            std::vector<DetectionResult> v{r, r2}; return b.vis_det_nv12(fr.dev(), v, opt);
        }
        case 1: {
            KeyPointsResult r; r.box = {fr.w * 0.05f, fr.h * 0.05f, fr.w * 0.90f, fr.h * 0.90f}; r.score = 0.90f;
            for (int i = 0; i < 17; ++i) r.keypoints.push_back(Point3f(fr.w * (0.12f + 0.045f * i), fr.h * (0.2f + 0.03f * (i % 5)), 0.9f));
            std::vector<KeyPointsResult> v{r}; return b.vis_pose_nv12(fr.dev(), v, opt);
        }
        case 2: {
            OCRResult rr;
            auto box = [&](float x, float y, float w, float h) {
                return std::array<int, 8>{ (int)(x * fr.w), (int)(y * fr.h), (int)((x + w) * fr.w), (int)(y * fr.h),
                                           (int)((x + w) * fr.w), (int)((y + h) * fr.h), (int)(x * fr.w), (int)((y + h) * fr.h) };
            };
            rr.boxes.push_back(box(0.1f, 0.1f, 0.6f, 0.15f)); rr.text.push_back("Hello World 2024");
            return b.vis_ocr_nv12(fr.dev(), rr, opt);
        }
        default: {
            ClassifyResult r; r.label_ids = {1, 3}; r.scores = {0.85f, 0.60f};
            return b.vis_cls_nv12(fr.dev(), r, opt, 5);
        }
    }
}

// ── 主机(CPU)图元绘制(与设备端相同场景的近似) ──
static void run_cpu(Frame& fr, CpuProcessorBackend& b, int scene,
                    const std::vector<uint8_t>& yy, const std::vector<uint8_t>& uu) {
    auto img = [&] { return fr.host_img(yy, uu); };
    switch (scene) {
        case 0:
            b.draw_rect_nv12(img(), fr.w * 0.10f, fr.h * 0.15f, fr.w * 0.45f, fr.h * 0.50f, 0, 158, 115, 2);
            b.draw_rect_nv12(img(), fr.w * 0.58f, fr.h * 0.20f, fr.w * 0.36f, fr.h * 0.45f, 240, 228, 66, 2);
            b.draw_text_nv12(img(), fr.w * 0.10f, fr.h * 0.15f - 20, "1: 0.92", 255, 255, 255, 2);
            b.draw_text_nv12(img(), fr.w * 0.58f, fr.h * 0.20f - 20, "3: 0.78", 255, 255, 255, 2);
            break;
        case 1: {
            b.draw_rect_nv12(img(), fr.w * 0.05f, fr.h * 0.05f, fr.w * 0.90f, fr.h * 0.90f, 230, 159, 0, 2);
            std::vector<Point3f> kpts;
            for (int i = 0; i < 17; ++i) kpts.emplace_back(fr.w * (0.12f + 0.045f * i), fr.h * (0.2f + 0.03f * (i % 5)), 0.9f);
            b.draw_points_nv12(img(), kpts, 0, 255, 0, 3);
            for (int i = 0; i + 1 < 17; ++i) {
                std::vector<Point2f> seg{Point2f(kpts[i].x, kpts[i].y), Point2f(kpts[i + 1].x, kpts[i + 1].y)};
                b.draw_polygon_nv12(img(), seg, 255, 0, 0, 1);
            }
            b.draw_text_nv12(img(), fr.w * 0.05f, fr.h * 0.05f - 20, "score: 0.90", 255, 255, 255, 2);
            break;
        }
        case 2: {
            std::vector<Point2f> box{Point2f(fr.w * 0.2f, fr.h * 0.1f), Point2f(fr.w * 0.7f, fr.h * 0.1f),
                                     Point2f(fr.w * 0.7f, fr.h * 0.25f), Point2f(fr.w * 0.2f, fr.h * 0.25f)};
            b.draw_polygon_nv12(img(), box, 245, 135, 66, 2);
            b.draw_text_nv12(img(), fr.w * 0.2f, fr.h * 0.1f - 20, "Hello World 2024", 255, 255, 255, 2);
            break;
        }
        default:
            for (int i = 0; i < 5; ++i)
                b.draw_text_nv12(img(), 8, 8 + i * 20, "1: 0.85 3: 0.60", 255, 255, 255, 2);
            break;
    }
}

static double best_double(double v, double* best) { return v < *best ? v : *best; }

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: test_bench_vis_holistic <input.jpg> [reps]\n"); return 2; }
    const cv::Mat rgb = cv::imread(argv[1]);
    if (rgb.empty()) { printf("FAIL read image\n"); return 2; }
    const int reps = argc >= 3 ? std::atoi(argv[2]) : 30;
    const int w = rgb.cols, h = rgb.rows;
    Frame fr; fr.w = w; fr.h = h;
    bgr_to_nv12(rgb, fr.y, fr.uv);
    fr.by.assign(fr.y.size(), 0); fr.buv.assign(fr.uv.size(), 0);
    if (cudaMalloc(&fr.d_y, fr.y.size()) != cudaSuccess || cudaMalloc(&fr.d_uv, fr.uv.size()) != cudaSuccess) {
        printf("FAIL cudaMalloc\n"); return 2;
    }
    fr.upload();
    CudaProcessorBackend cuda;
    CpuProcessorBackend cpu;
    VisionProcessorBackend::VisOptions opt; opt.threshold = 0.3f; opt.alpha = 0.35f;
    if (argc >= 4) opt.font_path = argv[3];
    const char* names[] = {"det", "pose", "ocr", "cls"};
    printf("== whole-scene vis draw bench: %dx%d, reps=%d ==\n", w, h, reps);
    // 预热：加载字体/驱动
    for (int s = 0; s < 4; ++s) { fr.upload(); run_dev(fr, cuda, s, opt); cudaDeviceSynchronize(); }
    // 帧级 D2H 拷贝（整帧，非场景）
    double d2h_best = 1e18;
    for (int r = 0; r < reps; ++r) d2h_best = best_double(fr.d2h_ms(), &d2h_best);
    printf("[D2H_copy] best=%.3fms (整帧 NV12 %.1fKB)\n", d2h_best, (fr.y.size() + fr.uv.size()) / 1024.0);
    for (int s = 0; s < 4; ++s) {
        double dev_best = 1e18, cpu_best = 1e18;
        for (int r = 0; r < reps; ++r) {
            fr.upload();
            auto t0 = std::chrono::high_resolution_clock::now();
            run_dev(fr, cuda, s, opt); cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            dev_best = best_double(std::chrono::duration<double, std::milli>(t1 - t0).count(), &dev_best);
            t0 = std::chrono::high_resolution_clock::now();
            run_cpu(fr, cpu, s, fr.y, fr.uv);
            t1 = std::chrono::high_resolution_clock::now();
            cpu_best = best_double(std::chrono::duration<double, std::milli>(t1 - t0).count(), &cpu_best);
        }
        const double host_total = d2h_best + cpu_best;
        printf("[%s] dev_draw=%.3fms  cpu_draw=%.3fms  host_total(D2H+cpu)=%.3fms  (D2H=%.3fms)\n",
               names[s], dev_best, cpu_best, host_total, d2h_best);
    }
    cudaFree(fr.d_y); cudaFree(fr.d_uv);
    printf("ALL_HOLISTIC_DONE\n");
    return 0;
}
