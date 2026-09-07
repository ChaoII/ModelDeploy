#pragma once
#include <string>
#include <vector>
#include "core/md_decl.h"

namespace modeldeploy::audio::tool {
// 逆文本归一化（Inverse Text Normalization, 口读 → 书面）。
// 把中文口读文本规整为书面数字形式（覆盖常见类别）：
//   一百二十三          -> 123
//   三点一四            -> 3.14
//   百分之五            -> 5%
//   四分之三            -> 3/4           （分数）
//   五元六角七分        -> 5.67元        （货币，含美元/日元/欧元等币种）
//   五十万元            -> 500000元
//   三公里 / 五百克     -> 3公里 / 500克 （度量，含米/公斤/吨等）
//   一三八零零...       -> 13800...      （电话/连续号码）
//   二零二四年五月九日  -> 2024年5月9日
//   五点三十分 / 五点半  -> 5:30
//   第八                -> 第8
//   壹佰贰拾叁          -> 123           （大写人民币数词归一）
// 有意保留模糊词（十几/几十/五一/三五/多/来 等）不做转换，避免误伤。
// 说明：EN/JA ITN 与依赖上下文的语义消歧（"三点"是时间还是小数）不在本轻量级
// 实现覆盖范围；此类需求应走 WeText/OpenFst 后端或 LLM。
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
