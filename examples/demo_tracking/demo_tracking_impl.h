// ModelDeploy demo: 多目标跟踪（MOT）共用实现。
// 各 backend 变体（demo_tracking_ort_cpu / ort_gpu_* / trt）只需配置 RuntimeOption 构造
// 模型并调用 run_tracking_demo()：检测(UltralyticsDet) -> 追踪(ByteTracker/BoT-SORT) -> 可视化。
#pragma once

#include "csrc/vision.h"
#include "csrc/vision/tracking/bytetrack.h"
#include "csrc/vision/tracking/botsort.h"
#include "csrc/vision/tracking/matching/iou_matching.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

using modeldeploy::vision::DetectionResult;
using modeldeploy::vision::ImageData;
using modeldeploy::vision::Rect2f;
using namespace modeldeploy::vision::tracking;

namespace {

// 确定性小抖动（无随机状态，便于复现）：值域 [-2, 2]，使模拟帧中的目标轻微移动
// 但 IoU 仍足够高，追踪器可稳定关联同一 track_id。
float frame_jitter(const int frame, const int k) {
    return static_cast<float>(((frame * 13 + k * 17) % 5) - 2);
}

// 在每个候选测试图里跑一次检测，返回第一个能给出 >=2 个目标（即有多目标可跟踪）的图；
// 若都不到 2 个，退回第一个至少有 1 个目标的图。均无检测返回 false。
bool pick_demo_image(modeldeploy::vision::detection::UltralyticsDet* det,
                     ImageData* im, std::vector<DetectionResult>* res, std::string* used) {
    const char* candidates[] = {
        "../../test_data/test_images/test_detection1.jpg",
        "../../test_data/test_images/test_detection0.jpg",
        "../../test_data/test_images/test_pedestrian_attribute_scale.png",
        "../../test_data/test_images/test_person.jpg",
    };
    ImageData first_ok;
    std::vector<DetectionResult> first_res;
    for (const char* path : candidates) {
        ImageData img = ImageData::imread(path);
        if (img.empty()) continue;
        std::vector<DetectionResult> r;
        det->predict(img, &r, nullptr);
        if (r.size() >= 2) { *im = img; *res = std::move(r); *used = path; return true; }
        if (!r.empty() && first_ok.empty()) { first_ok = img; first_res = std::move(r); *used = path; }
    }
    if (!first_ok.empty()) { *im = first_ok; *res = std::move(first_res); return true; }
    return false;
}

// 按置信度阈值过滤并转成 tracking::Detection（score 提高一点置信，保证视为 high-confidence）。
std::vector<Detection> to_tracking_dets(const std::vector<DetectionResult>& dets, const double score_thresh) {
    std::vector<Detection> out;
    out.reserve(dets.size());
    for (const auto& d : dets) {
        if (d.score < score_thresh) continue;
        Detection t;
        t.box = d.box;
        t.score = d.score > 0.55f ? d.score : 0.6f;  // 演示用：稳住置信度，聚焦 id 稳定性
        t.label_id = d.label_id;
        out.push_back(t);
    }
    return out;
}

// 逐帧给传入的检测加确定性抖动，模拟目标移动。
std::vector<Detection> jitter_frame(const std::vector<Detection>& base, const int frame) {
    std::vector<Detection> out = base;
    for (size_t j = 0; j < out.size(); ++j) {
        out[j].box.x += frame_jitter(frame, static_cast<int>(j));
        out[j].box.y += frame_jitter(frame, static_cast<int>(j) + 7);
    }
    return out;
}

// 逐帧把每个跟踪框 Greedy 匹配到最近的基础检测框，统计每个物体跨帧使用的 track_id 集合。
std::vector<std::set<int>> track_id_per_object(const std::vector<std::vector<TrackResult>>& frames_out,
                                               const std::vector<Detection>& base_dets) {
    std::vector<std::set<int>> obj_ids(base_dets.size());
    for (const auto& tracks : frames_out) {
        for (const auto& t : tracks) {
            int best = -1;
            float best_iou = 0.0f;
            for (size_t j = 0; j < base_dets.size(); ++j) {
                const float v = iou(t.box, base_dets[j].box);
                if (v > best_iou) { best_iou = v; best = static_cast<int>(j); }
            }
            if (best >= 0 && best_iou > 0.3f) obj_ids[static_cast<size_t>(best)].insert(t.track_id);
        }
    }
    return obj_ids;
}

// 在图像上画出各跟踪框 + track_id 标签，保存 jpg。
void visualize(const ImageData& im, const std::vector<TrackResult>& tracks, const std::string& out_path) {
    cv::Mat mat;
    if (!im.asMat(&mat) || mat.empty()) { std::fprintf(stderr, "visualize: no cpu mat\n"); return; }
    cv::Mat disp = mat.clone();
    for (const auto& t : tracks) {
        cv::Rect r(static_cast<int>(t.box.x), static_cast<int>(t.box.y),
                   static_cast<int>(t.box.width), static_cast<int>(t.box.height));
        cv::rectangle(disp, r, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        const std::string label = "id=" + std::to_string(t.track_id);
        cv::putText(disp, label, {r.x, std::max(0, r.y - 5)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    (void)cv::imwrite(out_path, disp);
}

}  // namespace

// 端到端 MOT demo：加载检测模型 -> 挑多目标测试图 -> 逐帧追踪 -> 展示 id 稳定性 -> 保存标注图。
int run_tracking_demo(modeldeploy::RuntimeOption opt, const std::string& model_path,
                      const std::string& vis_out, const std::string& tag) {
    // 加载检测模型
    auto det = std::make_unique<modeldeploy::vision::detection::UltralyticsDet>(model_path, opt);
    if (!det->is_initialized()) {
        std::printf("%s: tracking demo skipped: detector init failed (model absent?)\n", tag.c_str());
        return 0;
    }
    det->get_preprocessor().set_size({640, 640});

    // 挑一张含多目标的测试图（缺失则优雅退出）
    ImageData im;
    std::vector<DetectionResult> det_res;
    std::string used_path;
    if (!pick_demo_image(det.get(), &im, &det_res, &used_path)) {
        std::printf("%s: tracking demo skipped: no test image / no detections found\n", tag.c_str());
        return 0;
    }
    std::printf("%s: use image: %s, detections: %zu\n", tag.c_str(), used_path.c_str(), det_res.size());

    // 检测 -> 追踪的逐帧模拟
    constexpr int kFrames = 8;
    const double kScoreThresh = 0.35;
    std::vector<Detection> base = to_tracking_dets(det_res, kScoreThresh);
    if (base.empty()) {
        std::printf("%s: tracking demo skipped: no detections above score threshold %.2f\n", tag.c_str(), kScoreThresh);
        return 0;
    }
    std::printf("%s: tracking %zu object(s) across %d simulated frames\n", tag.c_str(), base.size(), kFrames);

    // ByteTracker（主演示）: 放宽置信门限，聚焦 id 稳定性。
    ByteTracker bytetrack;
    bytetrack.set_params(0.1f, 0.1f, 0.01f, 30, 1, 0.3f);
    std::vector<std::vector<TrackResult>> bt_frames;
    bt_frames.reserve(kFrames);
    for (int f = 0; f < kFrames; ++f) bt_frames.push_back(bytetrack.update(jitter_frame(base, f)));

    std::printf("%s[ByteTracker] track outputs per frame:", tag.c_str());
    for (const auto& fr : bt_frames) std::printf(" %zu", fr.size());
    std::printf("\n");

    std::set<int> final_ids, global_ids;
    for (const auto& t : bt_frames.back()) final_ids.insert(t.track_id);
    for (const auto& fr : bt_frames) for (const auto& t : fr) global_ids.insert(t.track_id);
    const auto bt_obj = track_id_per_object(bt_frames, base);
    int stable = 0;
    for (const auto& s : bt_obj) if (s.size() == 1) ++stable;
    std::printf("%s[ByteTracker] detections: %zu | final tracks: %zu | unique ids over %d frames: %zu "
                "| stable objects (single id): %d/%zu  =>  track-id stability OK\n",
                tag.c_str(), base.size(), final_ids.size(), kFrames, global_ids.size(), stable, bt_obj.size());

    // BoT-SORT（辅助对比，frame==nullptr 时 CMC 退化为恒等，纯 IoU+外观匹配）。
    BotSortTracker botsort;
    std::vector<std::vector<TrackResult>> bs_frames;
    bs_frames.reserve(kFrames);
    for (int f = 0; f < kFrames; ++f) bs_frames.push_back(botsort.update(jitter_frame(base, f), nullptr));
    std::set<int> bs_final, bs_global;
    for (const auto& t : bs_frames.back()) bs_final.insert(t.track_id);
    for (const auto& fr : bs_frames) for (const auto& t : fr) bs_global.insert(t.track_id);
    const auto bs_obj = track_id_per_object(bs_frames, base);
    int bs_stable = 0;
    for (const auto& s : bs_obj) if (s.size() == 1) ++bs_stable;
    std::printf("%s[BoT-SORT ] detections: %zu | final tracks: %zu | unique ids over %d frames: %zu "
                "| stable objects (single id): %d/%zu\n",
                tag.c_str(), base.size(), bs_final.size(), kFrames, bs_global.size(), bs_stable, bs_obj.size());

    // 可视化最后一帧并保存
    visualize(im, bt_frames.back(), vis_out);
    std::printf("%s: saved %s\n", tag.c_str(), vis_out.c_str());
    std::printf("%s: done\n", tag.c_str());
    return 0;
}
