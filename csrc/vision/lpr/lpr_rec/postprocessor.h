//
// Created by aichao on 2025/6/10.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/common/struct.h"


namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprRecPostprocessor {
    public:
        LprRecPostprocessor() = default;
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<LprResult>* results) const;

    private:
        std::vector<std::string> plate_color_list = {"黑色", "蓝色", "绿色", "白色", "黄色"};
        std::string plate_chr[78] = {
            "#", "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
            "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川",
            "贵", "云", "藏", "陕", "甘", "青", "宁",
            "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航", "危", "0", "1",
            "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
            "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z", "险", "品"
        };
    };
} // namespace modeldeploy
