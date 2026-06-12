#include "config.hpp"
#include <sstream>
#include <algorithm>
#include <cctype>

// ==================== TaskConfig 验证 ====================
bool TaskConfig::validate(std::string* err) const {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    if (input_url.empty()) return fail("input_url is empty");
    if (output_url.empty()) return fail("output_url is empty");
    if (models.empty()) return fail("at least one model required");
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        if (m.name.empty()) return fail("model[" + std::to_string(i) + "].name is empty");
        if (m.path.empty()) return fail("model[" + std::to_string(i) + "].path is empty");
        if (m.input_size.size() != 2) return fail("model[" + std::to_string(i) + "].input_size must be [w,h]");
        if (m.roi.size() != 4) return fail("model[" + std::to_string(i) + "].roi must be [x,y,w,h]");
        if (m.interval < 1) return fail("model[" + std::to_string(i) + "].interval must be >= 1");
    }
    return true;
}

void TaskConfig::print() const {
    std::cout << "Task: " << name << " (" << id << ")\n"
              << "  input:  " << input_url << "\n"
              << "  output: " << output_url << "\n"
              << "  models: " << models.size() << "\n";
    for (const auto& m : models) {
        std::cout << "    - " << m.name << " [" << m.type << "] "
                  << m.path << "  backend=" << m.backend
                  << " device=" << m.device
                  << " threshold=" << m.confidence_threshold
                  << " interval=" << m.interval;
        if (m.roi[2] > 0 && m.roi[3] > 0)
            std::cout << " roi=[" << m.roi[0] << "," << m.roi[1]
                      << "," << m.roi[2] << "," << m.roi[3] << "]";
        std::cout << "\n";
    }
}

// ==================== 最小 JSON 解析 ====================
// 实现简易递归下降解析器

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : s_(input), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        if (pos_ >= s_.size()) return {};
        char c = s_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
               s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }

    char peek() { skip_ws(); return pos_ < s_.size() ? s_[pos_] : '\0'; }
    char next() { return pos_ < s_.size() ? s_[pos_++] : '\0'; }

    void expect(char c) {
        if (next() != c) throw std::runtime_error(
            std::string("expected '") + c + "' at pos " + std::to_string(pos_ - 1));
    }

    JsonObject parse_object() {
        expect('{');
        JsonObject obj;
        if (peek() == '}') { next(); return obj; }
        while (true) {
            auto key = parse_string().as_string();
            expect(':');
            obj.set(key, parse());
            if (peek() == '}') { next(); break; }
            expect(',');
        }
        return obj;
    }

    JsonArray parse_array() {
        expect('[');
        JsonArray arr;
        if (peek() == ']') { next(); return arr; }
        while (true) {
            arr.push(parse());
            if (peek() == ']') { next(); break; }
            expect(',');
        }
        return arr;
    }

    JsonValue parse_string() {
        skip_ws();
        expect('"');
        std::string val;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            if (s_[pos_] == '\\') {
                ++pos_;
                if (pos_ < s_.size()) {
                    switch (s_[pos_]) {
                        case 'n': val += '\n'; break;
                        case 't': val += '\t'; break;
                        case 'r': val += '\r'; break;
                        case '\\': val += '\\'; break;
                        case '"': val += '"'; break;
                        default: val += s_[pos_]; break;
                    }
                    ++pos_;
                }
            } else {
                val += s_[pos_++];
            }
        }
        expect('"');
        return JsonValue(val);
    }

    JsonValue parse_number() {
        skip_ws();
        size_t start = pos_;
        if (s_[pos_] == '-') ++pos_;
        while (pos_ < s_.size() && std::isdigit(s_[pos_])) ++pos_;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            while (pos_ < s_.size() && std::isdigit(s_[pos_])) ++pos_;
        }
        double val = std::stod(s_.substr(start, pos_ - start));
        return JsonValue(val);
    }

    JsonValue parse_bool() {
        if (s_.substr(pos_, 4) == "true") { pos_ += 4; return JsonValue(true); }
        if (s_.substr(pos_, 5) == "false") { pos_ += 5; return JsonValue(false); }
        throw std::runtime_error("invalid bool at pos " + std::to_string(pos_));
    }

    JsonValue parse_null() {
        if (s_.substr(pos_, 4) == "null") { pos_ += 4; return {}; }
        throw std::runtime_error("invalid null at pos " + std::to_string(pos_));
    }
};

JsonValue parse_json(const std::string& input) {
    JsonParser parser(input);
    return parser.parse();
}

// ── JsonObject helpers ──
JsonValue* JsonObject::get(const std::string& key) {
    auto it = fields.find(key);
    return it != fields.end() ? &it->second : nullptr;
}
const JsonValue* JsonObject::get(const std::string& key) const {
    auto it = fields.find(key);
    return it != fields.end() ? &it->second : nullptr;
}
bool JsonObject::has(const std::string& key) const {
    return fields.find(key) != fields.end();
}
void JsonObject::set(const std::string& key, const JsonValue& v) {
    fields[key] = v;
}

// ── 路径读取 ──
// 路径格式: "models/0/name" 表示 root["models"][0]["name"]
static const JsonValue* resolve_path(const JsonValue& root, const std::string& path) {
    const JsonValue* cur = &root;
    size_t start = 0;
    while (start < path.size()) {
        size_t slash = path.find('/', start);
        std::string seg = path.substr(start, slash - start);
        if (seg.empty()) break;
        if (cur->is_object()) {
            auto* f = cur->as_object().get(seg);
            if (!f) return nullptr;
            cur = f;
        } else if (cur->is_array()) {
            int idx = std::stoi(seg);
            if (idx < 0 || (size_t)idx >= cur->as_array().size()) return nullptr;
            cur = &cur->as_array()[idx];
        } else {
            return nullptr;
        }
        start = (slash == std::string::npos) ? path.size() : slash + 1;
    }
    return cur;
}

std::string JsonValue::get_string(const JsonValue& root, const std::string& path,
                                   const std::string& def) {
    auto* v = resolve_path(root, path);
    if (!v || !v->is_string()) return def;
    return v->as_string();
}

double JsonValue::get_number(const JsonValue& root, const std::string& path, double def) {
    auto* v = resolve_path(root, path);
    if (!v || !v->is_number()) return def;
    return v->as_number();
}

int JsonValue::get_int(const JsonValue& root, const std::string& path, int def) {
    auto* v = resolve_path(root, path);
    if (!v || !v->is_number()) return def;
    return v->as_int();
}

bool JsonValue::get_bool(const JsonValue& root, const std::string& path, bool def) {
    auto* v = resolve_path(root, path);
    if (!v || !v->is_bool()) return def;
    return v->as_bool();
}

JsonArray JsonValue::get_array(const JsonValue& root, const std::string& path) {
    auto* v = resolve_path(root, path);
    if (!v || !v->is_array()) return {};
    return v->as_array();
}

// ── TaskConfig ←→ JsonValue ──
JsonValue task_config_to_json(const TaskConfig& cfg) {
    JsonObject root;
    root.set("id", JsonValue(cfg.id));
    root.set("name", JsonValue(cfg.name));
    root.set("input_url", JsonValue(cfg.input_url));
    root.set("output_url", JsonValue(cfg.output_url));

    // decoder
    JsonObject dec;
    dec.set("reconnect_delay_ms", JsonValue(cfg.decoder.reconnect_delay_ms));
    dec.set("max_reconnects", JsonValue(cfg.decoder.max_reconnects));
    dec.set("timeout_us", JsonValue(cfg.decoder.timeout_us));
    root.set("decoder", JsonValue(dec));

    // draw
    JsonObject drw;
    drw.set("show_label", JsonValue(cfg.draw.show_label));
    drw.set("show_score", JsonValue(cfg.draw.show_score));
    drw.set("font_path", JsonValue(cfg.draw.font_path));
    root.set("draw", JsonValue(drw));

    // models
    JsonArray models_arr;
    for (const auto& m : cfg.models) {
        JsonObject mo;
        mo.set("name", JsonValue(m.name));
        mo.set("type", JsonValue(m.type));
        mo.set("path", JsonValue(m.path));
        if (!m.rec_path.empty()) mo.set("rec_path", JsonValue(m.rec_path));
        mo.set("backend", JsonValue(m.backend));
        mo.set("device", JsonValue(m.device));
        mo.set("confidence_threshold", JsonValue(m.confidence_threshold));
        JsonArray sz;
        sz.push(JsonValue(m.input_size[0]));
        sz.push(JsonValue(m.input_size[1]));
        mo.set("input_size", JsonValue(sz));
        JsonArray roi;
        roi.push(JsonValue(m.roi[0])); roi.push(JsonValue(m.roi[1]));
        roi.push(JsonValue(m.roi[2])); roi.push(JsonValue(m.roi[3]));
        mo.set("roi", JsonValue(roi));
        mo.set("interval", JsonValue(m.interval));
        if (!m.labels.empty()) {
            JsonArray la;
            for (const auto& l : m.labels) la.push(JsonValue(l));
            mo.set("labels", JsonValue(la));
        }
        models_arr.push(JsonValue(mo));
    }
    root.set("models", JsonValue(models_arr));
    return JsonValue(root);
}

static std::vector<int> json_arr_to_int_vec(const JsonArray& arr) {
    std::vector<int> res;
    for (size_t i = 0; i < arr.size(); ++i)
        res.push_back(arr[i].as_int());
    return res;
}

TaskConfig task_config_from_json(const JsonValue& j) {
    TaskConfig cfg;
    auto& o = j.as_object();
    auto gs = [&](const std::string& k) -> std::string {
        auto* v = o.get(k);
        return (v && v->is_string()) ? v->as_string() : "";
    };
    cfg.id = gs("id");
    cfg.name = gs("name");
    cfg.input_url = gs("input_url");
    cfg.output_url = gs("output_url");

    // decoder
    if (auto* dec = o.get("decoder")) {
        auto& d = dec->as_object();
        if (auto* v = d.get("reconnect_delay_ms")) cfg.decoder.reconnect_delay_ms = v->as_int();
        if (auto* v = d.get("max_reconnects")) cfg.decoder.max_reconnects = v->as_int();
        if (auto* v = d.get("timeout_us")) cfg.decoder.timeout_us = v->as_int();
    }

    // draw
    if (auto* drw = o.get("draw")) {
        auto& d = drw->as_object();
        if (auto* v = d.get("show_label")) cfg.draw.show_label = v->as_bool();
        if (auto* v = d.get("show_score")) cfg.draw.show_score = v->as_bool();
        if (auto* v = d.get("font_path")) cfg.draw.font_path = v->as_string();
    }

    // models
    if (auto* marr = o.get("models")) {
        for (size_t i = 0; i < marr->as_array().size(); ++i) {
            auto& mo = marr->as_array()[i].as_object();
            ModelConfig m;
            auto g = [&](const std::string& k) -> std::string {
                auto* v = mo.get(k); return (v && v->is_string()) ? v->as_string() : "";
            };
            auto assign_str = [&](const std::string& key, std::string& target) {
                auto* v = mo.get(key);
                if (v && v->is_string()) target = v->as_string();
            };
            assign_str("name", m.name);
            assign_str("type", m.type);
            assign_str("path", m.path);
            assign_str("rec_path", m.rec_path);
            assign_str("backend", m.backend);
            assign_str("device", m.device);
            if (auto* v = mo.get("confidence_threshold")) m.confidence_threshold = (float)v->as_number();
            if (auto* v = mo.get("input_size")) m.input_size = json_arr_to_int_vec(v->as_array());
            if (auto* v = mo.get("roi")) m.roi = json_arr_to_int_vec(v->as_array());
            if (auto* v = mo.get("interval")) m.interval = v->as_int();
            if (auto* v = mo.get("labels")) {
                for (size_t j = 0; j < v->as_array().size(); ++j)
                    m.labels.push_back(v->as_array()[j].as_string());
            }
            cfg.models.push_back(m);
        }
    }
    return cfg;
}
