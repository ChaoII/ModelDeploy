//
// adapters.h —— 薄适配器：给真实 SDK 模型补上 AsyncModel<M>/make_model_handle 需要的契约。
//
// AsyncModel<M> 的调用约定：
//   predict(const ImageData&, M::result_type*, TimerArray*)
//   batch_predict(const vector<ImageData>&, vector<M::result_type>*, TimerArray*)
// 真实 SDK 模型（UltralyticsDet 等）大多已具备该签名，仅缺 result_type 类型别名；
// 由 ResultModel<Base, R> 补上即可。少数（classification 族）predict/batch_predict
// 缺少 TimerArray 形参，需 ClassifyAdapter 桥接（补默认形参并转发）。
//
// 注意：det 族的结果形态是 std::vector<DetectionResult>（predict 输出多框），故
// R = std::vector<vision::DetectionResult>；此时 batch_predict 输出
// std::vector<std::vector<DetectionResult>>，恰等于 AsyncModel 期望的 vector<R>。
#pragma once

#include "vision/detection/ultralytics_det.h"
#include "vision/classification/classification.h"

namespace modeldeploy::serving {

// Base：继承自某个真实 SDK 模型类；R：其 predict 输出的结果类型（det 族为
// std::vector<DetectionResult>）。仅补 result_type，其余（构造/方法）原样继承。
template <class Base, class R>
struct ResultModel : public Base {
    using result_type = R;
    using Base::Base;  // 继承构造函数，转发原有 (model_file, RuntimeOption[, ...])
};

// classification 族专用：Classification::predict/batch_predict 缺 TimerArray 形参，
// 与 AsyncModel 的 3 参调用不匹配。此处补上带默认 TimerArray 的中转（名称遮蔽基类
// 2 参版本；AsyncModel 经派生类型调用，命中 3 参版本）。不标 override（签名不同）。
template <class R>
class ClassifyAdapter : public modeldeploy::vision::classification::Classification {
public:
    using result_type = R;
    using Base = modeldeploy::vision::classification::Classification;
    using Base::Base;  // 继承构造函数 (model_file, RuntimeOption)

    bool predict(const modeldeploy::vision::ImageData& img, R* result, TimerArray* = nullptr) {
        return Base::predict(img, result);
    }

    bool batch_predict(const std::vector<modeldeploy::vision::ImageData>& imgs,
                       std::vector<R>* results, TimerArray* = nullptr) {
        return Base::batch_predict(imgs, results);
    }
};

}  // namespace modeldeploy::serving
