//
// ModelEntry —— 把 AsyncModel<M> 桥接为 JSON 类型擦除的 InferFn。
//
// make_model_handle<M> 把 M 的推理暴露为 ModelRepo::InferFn：
//   in["image"]        base64 → imdecode
//   in["image_path"]   路径 → imread
//   out["results"]     推理结果（R=std::string 时直接赋值；有 to_json 的类型亦可用）
//   out["duration_ms"] 推理耗时（毫秒）
//   out["model"]       name
//   out["params"]      in["params"] 原样回传（可选）
//
// 生命周期：make_model_handle 接收 std::unique_ptr<M>，内部转
// std::make_shared<AsyncModel<M>>(std::move(model), acfg) 并 start()（独占，无双删）；
// AsyncModel 由 shared_ptr 持有，InferFn 捕获该 shared_ptr 保证句柄存活期间模型不释放。
//
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <future>
#include "core/md_decl.h"
#include "serving/model_repo.h"
#include "pipeline/async_model.h"
#include "vision/common/image_data.h"

namespace modeldeploy::serving {

// 小的 header-only base64 编码：SDK 的 utils::base64_encode 未 dllexport，
// 而 make_model_handle 是模板、随可执行文件实例化，故此处自带实现（对应
// model_entry.cpp 里自带 base64_decode 的取舍）。
namespace detail {
inline std::string b64_encode(const std::vector<unsigned char>& data) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const auto a = data[i];
        const auto b = i + 1 < data.size() ? data[i + 1] : 0;
        const auto c = i + 2 < data.size() ? data[i + 2] : 0;
        out += chars[(a >> 2) & 0x3f];
        out += chars[((a << 4) | (b >> 4)) & 0x3f];
        out += i + 1 < data.size() ? chars[((b << 2) | (c >> 6)) & 0x3f] : '=';
        out += i + 2 < data.size() ? chars[c & 0x3f] : '=';
    }
    return out;
}
}  // namespace detail

// 解析输入 JSON → ImageData（in["image"] base64 / in["image_path"] 路径）。
// 二者皆无或解码/读取失败返回空图并填 err（供 HTTP 400）。
MODELDEPLOY_CXX_EXPORT
vision::ImageData image_from_json(const nlohmann::json& in, std::string* err);

// 模板工厂：把 M 的推理封装成 JSON InferFn。M 需满足异步契约
// （AsyncModel<M> 可用；M::result_type 存在；predict/batch_predict 如 FakeModel）。
//
// vis（可选）：按族把【原图 + 推理结果】渲染成 SDK 标注图（调用 vision::vis_*）。
// 非空标注图会编码为 base64 写进 out["image_b64"]（及 image_w/image_h）。
template <typename M>
ModelHandle make_model_handle(
    std::string name, std::unique_ptr<M> model,
    const pipeline::AsyncModelConfig& acfg = {},
    std::function<vision::ImageData(vision::ImageData&, const typename M::result_type&)> vis =
        {}) {
    auto async = std::make_shared<pipeline::AsyncModel<M>>(std::move(model), acfg);
    std::string start_err;
    if (!async->start(&start_err)) {
        throw std::runtime_error("make_model_handle: AsyncModel start failed: " + start_err);
    }

    ModelHandle h;
    h.name = name;
    h.status = ModelStatus::Ready;
    const std::string model_name = name;

    h.infer = [async, model_name, vis](const nlohmann::json& in, nlohmann::json* out,
                                       std::string* err,
                                       std::chrono::milliseconds timeout) -> InferStatus {
        modeldeploy::vision::ImageData img = image_from_json(in, err);
        if (img.empty()) return InferStatus::Failed;

        auto t0 = std::chrono::steady_clock::now();
        typename M::result_type res;
        std::future<typename M::result_type> fut;
        try {
            fut = async->predict_async(img);
        } catch (const std::exception& e) {
            if (err) *err = std::string("inference failed: ") + e.what();
            return InferStatus::Failed;
        }
        // 超时：放弃等待返回 Timeout；AsyncModel 用 promise 基 future，析构不阻塞，
        // worker 继续跑完（结果无人取，无泄漏）。timeout<=0 表示不限时。
        if (timeout.count() > 0 && fut.wait_for(timeout) == std::future_status::timeout) {
            return InferStatus::Timeout;
        }
        try {
            res = fut.get();
        } catch (const std::exception& e) {
            if (err) *err = std::string("inference failed: ") + e.what();
            return InferStatus::Failed;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (out) {
            (*out)["results"] = res;
            (*out)["duration_ms"] = ms;
            (*out)["model"] = model_name;
            if (in.contains("params")) (*out)["params"] = in["params"];
            // 可视化可选：仅当请求显式 visualize=true 时渲染标注图（base64 JPEG）。
            if (vis && in.value("visualize", false)) {
                try {
                    modeldeploy::vision::ImageData viz = vis(img, res);
                    if (!viz.empty()) {
                        auto bytes = modeldeploy::vision::ImageData::imencode(viz, ".jpg");
                        (*out)["image_b64"] = detail::b64_encode(bytes);
                        (*out)["image_w"] = viz.width();
                        (*out)["image_h"] = viz.height();
                    }
                } catch (const std::exception&) {
                    // 可视化失败不影响核心推理结果
                }
            }
        }
        return InferStatus::Ok;
    };
    return h;
}

}  // namespace modeldeploy::serving
