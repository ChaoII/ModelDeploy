#pragma once
#include <string>
#include <vector>
#include "core/md_decl.h"

namespace modeldeploy::audio::tool {
// 逆文本归一化（Inverse Text Normalization, 口读 → 书面）。
// 把中文口读文本中的数字/小数/百分数/年份/日期/时间/序数 规整为书面数字形式：
//   一百二十三          -> 123
//   三点一四            -> 3.14
//   百分之五            -> 5%
//   二零二四年五月九日   -> 2024年5月9日
//   五点三十分 / 五点半  -> 5:30
//   第八                -> 第8
// 有意保留模糊词（十几/几十/五一/三五/多/来 等）不做转换，避免误伤。
class MODELDEPLOY_CXX_EXPORT InverseTextNormalizer {
public:
    InverseTextNormalizer() = default;
    std::string normalize(const std::string& text) const;
    std::wstring normalize(const std::wstring& text) const;

private:
    // 仅处理一个不含 万/亿 的片段（十/百/千）
    static long long parse_low(const std::wstring& s);
    // 处理含 万/亿 的完整数字串
    static long long parse_cn_number(const std::wstring& s);
    // 数字串 -> 书面：含 万/亿 时保留单位分组，否则输出纯阿拉伯数字
    static std::wstring render_cn_number(const std::wstring& s);
    // 是否可安全当作数字词转换（含单位或长度>=2）
    static bool convertible_run(const std::wstring& run);
};
} // namespace modeldeploy::audio::tool
