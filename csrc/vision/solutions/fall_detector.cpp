#include "vision/solutions/fall_detector.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::vision::solution {
namespace {
constexpr float kPi = 3.14159265358979323846f;
} // namespace

float FallDetector::angle(const Point3f& a, const Point3f& b, const Point3f& c) {
    const float abx = a.x - b.x, aby = a.y - b.y;
    const float cbx = c.x - b.x, cby = c.y - b.y;
    const float dot = abx * cbx + aby * cby;
    const float n1 = std::sqrt(abx * abx + aby * aby);
    const float n2 = std::sqrt(cbx * cbx + cby * cby);
    if (n1 <= 0 || n2 <= 0) return 180.0f;
    const float cosv = std::max(-1.0f, std::min(1.0f, dot / (n1 * n2)));
    return std::acos(cosv) * 180.0f / kPi;
}

FallResult FallDetector::update(const std::vector<KeyPointsResult>& persons) {
    FallResult result;
    result.state = state_;
    result.confidence = confidence_;
    if (persons.empty()) {
        return result;
    }

    // 对每个目标计算躯干倾角与重心（髋中点 y）平滑高度，取最大倾角目标作为被跟踪者。
    if (last_heights_.size() != persons.size()) {
        last_heights_.resize(persons.size(), 0.f);
    }
    float best_tilt = -1.0f;
    float smoothed_h = 0.0f;
    for (size_t i = 0; i < persons.size(); ++i) {
        const auto& kp = persons[i].keypoints;
        // 缺肩/髋关键点（或置信度为 0）则跳过该目标
        if (kp.size() < 17) continue;
        if (kp[kLeftShoulder].z <= 0 || kp[kRightShoulder].z <= 0 ||
            kp[kLeftHip].z <= 0 || kp[kRightHip].z <= 0) {
            continue;
        }
        const Point3f sh_mid{(kp[kLeftShoulder].x + kp[kRightShoulder].x) * 0.5f,
                             (kp[kLeftShoulder].y + kp[kRightShoulder].y) * 0.5f, 0.f};
        const Point3f hp_mid{(kp[kLeftHip].x + kp[kRightHip].x) * 0.5f,
                             (kp[kLeftHip].y + kp[kRightHip].y) * 0.5f, 0.f};
        const float raw_h = hp_mid.y;
        float prev = last_heights_[i];
        float sm = (prev <= 0.0f) ? raw_h : kHeightAlpha * raw_h + (1.0f - kHeightAlpha) * prev;
        last_heights_[i] = sm;

        // 竖直参考点：肩中点正下方（同 x，y 下移 100px）
        const Point3f vref{sh_mid.x, sh_mid.y + 100.0f, 0.f};
        const float tilt = angle(hp_mid, sh_mid, vref);
        if (tilt > best_tilt) {
            best_tilt = tilt;
            smoothed_h = sm;
        }
    }
    if (best_tilt < 0.0f) {
        return result;  // 本帧无有效目标，保持状态
    }

    // 站立期间持续以低通更新基准重心高度
    if (state_ == FallState::Standing) {
        baseline_h_ = (baseline_h_ <= 0.0f) ? smoothed_h
                                            : kHeightAlpha * smoothed_h + (1.0f - kHeightAlpha) * baseline_h_;
    }
    const float drop_ratio =
        (baseline_h_ > 0.0f) ? (baseline_h_ - smoothed_h) / (baseline_h_ + 1.0f) : 0.0f;

    const float tilt = best_tilt;
    if (tilt > kFallTiltDeg) {
        ++t_suspect_;
    } else if (tilt < kStandTiltDeg) {
        t_suspect_ = 0;
    }

    if (t_suspect_ >= kSuspectFrames) {
        state_ = FallState::Fallen;
    } else if (tilt > kFallTiltDeg) {
        state_ = FallState::PreFall;
    } else if (tilt < kStandTiltDeg) {
        state_ = FallState::Standing;
        t_suspect_ = 0;
    }

    // 置信度：倾角越大、重心越低越可信（Fallen/PreFall 时给出）
    float conf = 0.0f;
    if (state_ == FallState::Fallen || state_ == FallState::PreFall) {
        const float tilt_norm =
            std::clamp((tilt - kFallTiltDeg) / (90.0f - kFallTiltDeg), 0.0f, 1.0f);
        const float drop_norm = std::clamp(drop_ratio / 0.5f, 0.0f, 1.0f);
        conf = 0.7f * tilt_norm + 0.3f * drop_norm;
    }
    confidence_ = conf;
    result.state = state_;
    result.confidence = confidence_;
    return result;
}
} // namespace modeldeploy::vision::solution
