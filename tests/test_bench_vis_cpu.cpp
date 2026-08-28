// CPU 端 NV12 就地图元绘制性能基准：复用与 render_vis_gpu 相同的合成场景，
// 用 CpuProcessorBackend 的低层图元(rect/polygon/points/text) + CPU overlay 复现，逐场景计时。
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"
#include "vision/processors/cpu/cpu_processor_backend.h"

using namespace modeldeploy::vision;

struct Frame {
    int w = 0, h = 0;
    std::vector<uint8_t> y, uv;
    ImageData img;
    ImageData& make() { return img; }
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

// 每场景返回相对该场景的绘制耗时(ms)；cv::Mat 仅用于构造，不做时间统计。
static double exec(CpuProcessorBackend& b, Frame& fr, int scene) {
    CpuProcessorBackend::VisOptions opt;
    const auto t0 = std::chrono::high_resolution_clock::now();
    switch (scene) {
        case 0: {  // det：2 框 + 标签文本
            b.draw_rect_nv12(fr.make(), fr.w * 0.10f, fr.h * 0.15f, fr.w * 0.45f, fr.h * 0.50f, 0, 158, 115, 2);
            b.draw_rect_nv12(fr.make(), fr.w * 0.58f, fr.h * 0.20f, fr.w * 0.36f, fr.h * 0.45f, 240, 228, 66, 2);
            b.draw_text_nv12(fr.make(), fr.w * 0.10f, fr.h * 0.15f - 20, "1: 0.92", 255, 255, 255, 2);
            b.draw_text_nv12(fr.make(), fr.w * 0.58f, fr.h * 0.20f - 20, "3: 0.78", 255, 255, 255, 2);
            break;
        }
        case 1: {  // pose：框 + 17 关键点 + 19 肢干线
            b.draw_rect_nv12(fr.make(), fr.w * 0.05f, fr.h * 0.05f, fr.w * 0.90f, fr.h * 0.90f, 230, 159, 0, 2);
            std::vector<Point3f> kpts;
            for (int i = 0; i < 17; ++i)
                kpts.emplace_back(fr.w * (0.2f + 0.035f * i), fr.h * (0.2f + 0.03f * (i % 5)), 0.9f);
            b.draw_points_nv12(fr.make(), kpts, 0, 255, 0, 3);
            for (int i = 0; i + 1 < 17; ++i) {
                std::vector<Point2f> seg{Point2f(kpts[i].x, kpts[i].y), Point2f(kpts[i + 1].x, kpts[i + 1].y)};
                b.draw_polygon_nv12(fr.make(), seg, 255, 0, 0, 1);
            }
            b.draw_text_nv12(fr.make(), fr.w * 0.05f, fr.h * 0.05f - 20, "score: 0.90", 255, 255, 255, 2);
            break;
        }
        case 2: {  // ocr：文本框 + 文本
            std::vector<Point2f> box{Point2f(fr.w * 0.2f, fr.h * 0.1f), Point2f(fr.w * 0.7f, fr.h * 0.1f),
                                     Point2f(fr.w * 0.7f, fr.h * 0.25f), Point2f(fr.w * 0.2f, fr.h * 0.25f)};
            b.draw_polygon_nv12(fr.make(), box, 245, 135, 66, 2);
            b.draw_text_nv12(fr.make(), fr.w * 0.2f, fr.h * 0.1f - 20, "Hello World 2024", 255, 255, 255, 2);
            break;
        }
        case 3: {  // cls / attr：文本若干行
            for (int i = 0; i < 5; ++i)
                b.draw_text_nv12(fr.make(), 8, 8 + i * 20, "1: 0.85 3: 0.60", 255, 255, 255, 2);
            break;
        }
        default: break;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: test_bench_vis_cpu <input.jpg> [reps]\n"); return 2; }
    const cv::Mat rgb = cv::imread(argv[1]);
    if (rgb.empty()) { printf("FAIL read image\n"); return 2; }
    const int reps = argc >= 3 ? std::atoi(argv[2]) : 10;
    const int w = rgb.cols, h = rgb.rows;
    Frame fr; fr.w = w; fr.h = h;
    bgr_to_nv12(rgb, fr.y, fr.uv);
    const ImageData::Plane pl[2] = {{fr.y.data(), w}, {fr.uv.data(), w}};
    fr.img = ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU);
    CpuProcessorBackend backend;
    const char* names[] = {"det", "pose", "ocr", "cls_attr"};
    for (int s = 0; s < 4; ++s) {
        double best = 1e18, sum = 0;
        for (int r = 0; r < reps; ++r) {
            double ms = exec(backend, fr, s);
            if (ms < best) best = ms;
            sum += ms;
        }
        printf("[%s] cpu best=%.3fms avg=%.3fms (reps=%d)\n", names[s], best, sum / reps, reps);
    }
    printf("ALL_CPU_DONE image=%dx%d\n", w, h);
    return 0;
}
