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
    std::vector<ModelConfig> models;
    DecoderConfig decoder;
    DrawConfig draw;

    [[nodiscard]] bool validate(std::string* err = nullptr) const;
    void print() const;
};

// ==================== JSON 序列化 ====================
json task_config_to_json(const TaskConfig& cfg);
TaskConfig task_config_from_json(const json& j);
