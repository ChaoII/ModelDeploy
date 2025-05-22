//
// Created by aichao on 2025/5/22.
//

#include <random>
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    cv::Scalar get_random_color() {
        std::random_device rd; // 获取随机数种子
        std::mt19937 gen(rd()); // 使用Mersenne Twister算法生成随机数
        std::uniform_int_distribution dis(0, 255); // 定义随机数范围为1到255
        return {
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen))
        };
    }
}
