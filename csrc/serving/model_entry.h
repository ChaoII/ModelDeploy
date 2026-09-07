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

#include <memory>
#include <stdexcept>
#include <string>
#include <chrono>
#include "core/md_decl.h"
#include "serving/model_repo.h"
#include "pipeline/async_model.h"
#include "vision/common/image_data.h"

namespace modeldeploy::serving {

// 解析输入 JSON → ImageData（in["image"] base64 / in["image_path"] 路径）。
// 二者皆无或解码/读取失败返回空图并填 err（供 HTTP 400）。
MODELDEPLOY_CXX_EXPORT
vision::ImageData image_from_json(const nlohmann::json& in, std::string* err);

// 模板工厂：把 M 的推理封装成 JSON InferFn。M 需满足异步契约
// （AsyncModel<M> 可用；M::result_type 存在；predict/batch_predict 如 FakeModel）。
template <typename M>
ModelHandle make_model_handle(std::string name, std::unique_ptr<M> model,
                              const pipeline::AsyncModelConfig& acfg = {}) {
    auto async = std::make_shared<pipeline::AsyncModel<M>>(std::move(model), acfg);
    std::string start_err;
    if (!async->start(&start_err)) {
        throw std::runtime_error("make_model_handle: AsyncModel start failed: " + start_err);
    }

    ModelHandle h;
    h.name = name;
    h.ready = true;
    const std::string model_name = name;

    h.infer = [async, model_name](const nlohmann::json& in, nlohmann::json* out,
                                  std::string* err) {
        modeldeploy::vision::ImageData img = image_from_json(in, err);
        if (img.empty()) return false;

        auto t0 = std::chrono::steady_clock::now();
        typename M::result_type res;
        try {
            auto fut = async->predict_async(img);
            res = fut.get();
        } catch (const std::exception& e) {
            if (err) *err = std::string("inference failed: ") + e.what();
            return false;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (out) {
            (*out)["results"] = res;
            (*out)["duration_ms"] = ms;
            (*out)["model"] = model_name;
            if (in.contains("params")) (*out)["params"] = in["params"];
        }
        return true;
    };
    return h;
}

}  // namespace modeldeploy::serving
