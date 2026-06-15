#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// ==================== 模型配置 ====================
struct ModelConfig {
    std::string name;
    std::string type = "detection";
    std::string path;
    std::string rec_path;
    std::string backend = "ort";
    std::string device = "gpu";
    float confidence_threshold = 0.5f;
    std::vector<int> input_size = {640, 640};
    std::vector<int> roi = {0, 0, 0, 0};
    int interval = 1;
    std::vector<std::string> labels;
};

// ==================== 解码器配置 ====================
struct DecoderConfig {
    int reconnect_delay_ms = 5000;
    int max_reconnects = 10;
    int timeout_us = 10000000;
    std::string rtsp_transport = "tcp";   // tcp / udp
    std::string hw_accel = "cuda";        // cuda / none
};

// ==================== 编码器配置 ====================
struct EncoderConfig {
    int fps = 25;
    int bitrate_kbps = 2500;              // 平均码率 (kbps)，预览路 2.5Mbps 足够
    int gop = 12;                         // 关键帧间隔（约 0.5 秒/I 帧，flv.js 快速首帧）
    std::string codec = "auto";           // auto / libx264 / h264_nvenc
    std::string preset = "ultrafast";     // x264: ultrafast..veryslow ; nvenc: p1..p7
    std::string tune = "zerolatency";     // x264 only
    std::string format = "auto";          // auto / rtsp / rtmp / flv / mp4
    int max_b_frames = 0;
    bool low_latency = true;
};

// ==================== 绘制配置 ====================
struct DrawConfig {
    bool show_label = true;
    bool show_score = true;
    std::string font_path;
};

// ==================== 单路任务配置 ====================
struct TaskConfig {
    std::string id;
    std::string name;
    std::string input_url;
    std::string output_url;
    std::string preview_url;              // 浏览器拉流地址（HTTP-FLV 等）
    std::vector<ModelConfig> models;
    DecoderConfig decoder;
    EncoderConfig encoder;
    DrawConfig draw;

    [[nodiscard]] bool validate(std::string* err = nullptr) const;
    void print() const;
};

// ==================== JSON 序列化 ====================
json task_config_to_json(const TaskConfig& cfg);
TaskConfig task_config_from_json(const json& j);
