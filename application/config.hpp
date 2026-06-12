#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>

// ==================== 模型配置 ====================
struct ModelConfig {
    std::string name;
    std::string type = "detection";       // detection | classification | face_detection | lpr_pipeline | pose
    std::string path;
    std::string rec_path;                 // for pipeline models (e.g. LPR)
    std::string backend = "ort";          // ort | trt | mnn
    std::string device = "gpu";           // gpu | cpu
    float confidence_threshold = 0.5f;
    std::vector<int> input_size = {640, 640};
    std::vector<int> roi = {0, 0, 0, 0}; // x,y,w,h; 0,0,0,0 = full frame
    int interval = 1;                     // run every N frames (1 = every frame)
    std::vector<std::string> labels;      // empty = all labels
};

// ==================== 解码器配置 ====================
struct DecoderConfig {
    int reconnect_delay_ms = 5000;
    int max_reconnects = 10;
    int timeout_us = 10000000;            // 10s
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

    // 内部验证
    [[nodiscard]] bool validate(std::string* err = nullptr) const;
    void print() const;
};

// ==================== JSON 序列化 ====================
#ifdef SURVEILLANCE_USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#else
// 最小化 JSON 解析（仅满足需要，不引入 nlohmann）
// 使用简单的递归下降解析，支持对象/数组/字符串/数字/布尔
class JsonValue;
class JsonObject {
public:
    std::map<std::string, JsonValue> fields;
    JsonValue* get(const std::string& key);
    const JsonValue* get(const std::string& key) const;
    bool has(const std::string& key) const;
    void set(const std::string& key, const JsonValue& v);
};

class JsonArray {
public:
    std::vector<JsonValue> items;
    size_t size() const { return items.size(); }
    const JsonValue& operator[](size_t i) const { return items[i]; }
    JsonValue& operator[](size_t i) { return items[i]; }
    void push(const JsonValue& v) { items.push_back(v); }
};

class JsonValue {
public:
    enum Type { NUL, OBJ, ARR, STR, NUM, BOOL };
    Type type_ = NUL;
    JsonObject obj_;
    JsonArray arr_;
    std::string str_;
    double num_ = 0;
    bool bool_ = false;

    JsonValue() : type_(NUL) {}
    JsonValue(const JsonObject& o) : type_(OBJ), obj_(o) {}
    JsonValue(const JsonArray& a) : type_(ARR), arr_(a) {}
    JsonValue(const std::string& s) : type_(STR), str_(s) {}
    JsonValue(double n) : type_(NUM), num_(n) {}
    JsonValue(bool b) : type_(BOOL), bool_(b) {}
    JsonValue(int n) : type_(NUM), num_(static_cast<double>(n)) {}

    bool is_null() const { return type_ == NUL; }
    bool is_object() const { return type_ == OBJ; }
    bool is_array() const { return type_ == ARR; }
    bool is_string() const { return type_ == STR; }
    bool is_number() const { return type_ == NUM; }
    bool is_bool() const { return type_ == BOOL; }

    const JsonObject& as_object() const { return obj_; }
    JsonObject& as_object() { return obj_; }
    const JsonArray& as_array() const { return arr_; }
    JsonArray& as_array() { return arr_; }
    const std::string& as_string() const { return str_; }
    double as_number() const { return num_; }
    bool as_bool() const { return bool_; }

    // 辅助：数字→int
    int as_int() const { return static_cast<int>(num_); }

    // 辅助：从 object 中按路径读取
    static std::string get_string(const JsonValue& root, const std::string& path,
                                   const std::string& def = "");
    static double get_number(const JsonValue& root, const std::string& path, double def = 0);
    static int get_int(const JsonValue& root, const std::string& path, int def = 0);
    static bool get_bool(const JsonValue& root, const std::string& path, bool def = false);
    static JsonArray get_array(const JsonValue& root, const std::string& path);
};

// 解析 JSON 字符串 → JsonValue
JsonValue parse_json(const std::string& input);

// ── TaskConfig ←→ JsonValue ──
JsonValue task_config_to_json(const TaskConfig& cfg);
TaskConfig task_config_from_json(const JsonValue& j);

#endif // SURVEILLANCE_USE_NLOHMANN_JSON
