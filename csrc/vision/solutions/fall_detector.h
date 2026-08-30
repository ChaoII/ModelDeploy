#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"
#include "vision/solutions/solution_base.h"

namespace modeldeploy::vision::solution {
    // COCO 17 关键点索引（仓库 vis_pose.cpp 约定，0=鼻，无颈点；左右肩 5/6，左右髋 11/12）
    enum FallKeyIndex : int {
        kLeftShoulder = 5,
        kRightShoulder = 6,
        kLeftHip = 11,
        kRightHip = 12,
    };

    enum class FallState : int {
        Standing = 0,
        PreFall = 1,
        Fallen = 2,
    };

    struct MODELDEPLOY_CXX_EXPORT FallResult {
        FallState state{FallState::Standing};
        float confidence{0.f};
    };

    /*! @brief 跌倒检测姿态规则方案（零训练，复用 UltralyticsPose 输出的 KeyPointsResult）
     *
     * 依据躯干中轴（髋中点→肩中点）相对竖直方向倾角 + 重心（髋中点纵坐标）下降的
     * 阈值状态机判定：倾角持续 > 60° 达 N 帧进入 Fallen，倾角回落 < 30° 复位 Standing。
     */
    class MODELDEPLOY_CXX_EXPORT FallDetector : public SolutionBase {
    public:
        static constexpr float kFallTiltDeg = 60.0f;
        static constexpr float kStandTiltDeg = 30.0f;
        static constexpr int kSuspectFrames = 3;
        static constexpr float kHeightAlpha = 0.4f;

        FallDetector() = default;

        void reset() override {
            last_heights_.clear();
            baseline_h_ = 0.f;
            t_suspect_ = 0;
            state_ = FallState::Standing;
            confidence_ = 0.f;
        }

        FallResult update(const std::vector<KeyPointsResult>& persons);

        /// 三点二维投影夹角（度），与 WorkoutMonitor::angle 同款几何公式
        static float angle(const Point3f& a, const Point3f& b, const Point3f& c);

    private:
        std::vector<float> last_heights_;   // 每目标最近重心（髋中点 y）指数平滑值
        float baseline_h_{0.f};             // 站立期间平滑后的基准重心高度
        int t_suspect_{0};                  // 持续"疑似"帧数
        FallState state_{FallState::Standing};
        float confidence_{0.f};
    };
} // namespace modeldeploy::vision::solution
