#pragma once
#include <array>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::tracking {
    // SORT/ByteTrack-style 8-dim constant-velocity Kalman filter.
    // State: [x, y, a, h, vx, vy, va, vh] (constant velocity) where (x, y) is the
    // center, a = width / height aspect ratio, h = height.
    class MODELDEPLOY_CXX_EXPORT KalmanFilter {
    public:
        KalmanFilter();

        void init(const Rect2f& box);
        void predict();
        void update(const Rect2f& box);
        Rect2f get_state() const;
        std::array<double, 8> get_covariance_diag() const;

        static constexpr double chi2inv95[10] = {
            3.841458820694124, 5.991464547107979, 7.814727903251179,
            9.487729036781154, 11.070497693516351, 12.591587243743977,
            14.067140449340169, 15.507313055865453, 16.918977604605994,
            18.307037615984347};

    private:
        std::array<double, 8> mean_{};
        std::array<std::array<double, 8>, 8> covariance_{};
        bool initialized_{false};

        static std::array<double, 4> convert_rect_to_xyah(const Rect2f& box);
        static Rect2f convert_xyah_to_rect(const std::array<double, 4>& xyah);
    };
}
