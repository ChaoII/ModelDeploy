//
// Created by aichao on 2025/8/2.
// 模型输入输出信息表的统一样式助手（各后端复用，风格与 ORT/TRT 一致）。
//

#pragma once

#include <string>
#include <vector>
#include <tabulate/tabulate.hpp>
#include "runtime/backends/backend.h"

namespace modeldeploy {

    // 由输入/输出描述构建一个 tabulate 信息表，可直接 << 到 MD_LOG_INFO。
    // 列：Type / Index / Name / Data Type / Shape。
    inline tabulate::Table build_io_table(const std::vector<TensorInfo>& inputs,
                                          const std::vector<TensorInfo>& outputs) {
        tabulate::Table t;
        t.format().font_color(tabulate::Color::yellow)
                  .border_color(tabulate::Color::blue)
                  .corner_color(tabulate::Color::blue);
        t.add_row({"Type", "Index", "Name", "Data Type", "Shape"});
        t[0].format().font_style({tabulate::FontStyle::bold});
        for (size_t i = 0; i < inputs.size(); ++i) {
            t.add_row({"Input", std::to_string(i), inputs[i].name,
                       datatype_to_string(inputs[i].dtype), vector_to_string(inputs[i].shape)});
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            t.add_row({"Output", std::to_string(i), outputs[i].name,
                       datatype_to_string(outputs[i].dtype), vector_to_string(outputs[i].shape)});
        }
        return t;
    }

} // namespace modeldeploy
