//
// ModelEntry —— 非模板部分：JSON 输入 → ImageData 的解析。
// 模板工厂 make_model_handle 定义在 model_entry.h（需随 M 实例化）。
//
// base64：SDK 的 utils::base64_decode 未加 dllexport（DLL 不导出给外部），
// 且本任务提交范围限定在 model_entry.h/.cpp + test_serving.cpp，故此处内置
// 一个自包含（纯标准库）的 base64 解码，仅供本模块使用。
//
#include "serving/model_entry.h"

#include <string>
#include <vector>

namespace modeldeploy::serving {

namespace {

std::vector<unsigned char> b64_decode(const std::string& s) {
    static const std::string tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> out;
    int val = 0;
    int bits = -8;
    for (unsigned char c : s) {
        if (c == '=') break;
        std::string::size_type pos = tbl.find(static_cast<char>(c));
        if (pos == std::string::npos) break;  // 非法字符 → 停止解码
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<unsigned char>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

}  // namespace

MODELDEPLOY_CXX_EXPORT
vision::ImageData image_from_json(const nlohmann::json& in, std::string* err) {
    if (in.contains("image")) {
        if (!in["image"].is_string()) {
            if (err) *err = "'image' must be a base64 string";
            return vision::ImageData();
        }
        auto bytes = b64_decode(in["image"].get<std::string>());
        vision::ImageData img;
        try {
            img = vision::ImageData::imdecode(bytes);
        } catch (const std::exception&) {
            img = vision::ImageData();
        }
        if (img.empty() || img.width() <= 0 || img.height() <= 0) {
            if (err) *err = "invalid 'image': base64 decode / imdecode failed";
            return vision::ImageData();
        }
        return img;
    }
    if (in.contains("image_path")) {
        if (!in["image_path"].is_string()) {
            if (err) *err = "'image_path' must be a string";
            return vision::ImageData();
        }
        vision::ImageData img;
        try {
            img = vision::ImageData::imread(in["image_path"].get<std::string>());
        } catch (const std::exception&) {
            img = vision::ImageData();
        }
        if (img.empty()) {
            if (err) *err = "invalid 'image_path': imread failed";
            return vision::ImageData();
        }
        return img;
    }
    {
        if (err) *err = "missing 'image' or 'image_path'";
        return vision::ImageData();
    }
}

}  // namespace modeldeploy::serving
