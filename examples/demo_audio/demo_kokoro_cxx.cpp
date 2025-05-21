//
// Created by aichao on 2025/5/20.
//


#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "csrc/audio/tts/kokoro.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int32_t main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const std::string kokoro_onnx = "../../test_data/test_models/kokoro-multi-lang-v1_1/model.onnx";
    const std::string tokens = "../../test_data/test_models/kokoro-multi-lang-v1_1/tokens.txt";
    const std::vector<std::string> lexicons = {
        "../../test_data/test_models/kokoro-multi-lang-v1_1/lexicon-us-en.txt",
        "../../test_data/test_models/kokoro-multi-lang-v1_1/lexicon-zh.txt"
    };
    const std::string voice_bin = "../../test_data/test_models/kokoro-multi-lang-v1_1/voices.bin";
    const std::string jieba_dir = "../../test_data/test_models/kokoro-multi-lang-v1_1/dict/";
    auto kokoro = modeldeploy::Kokoro(kokoro_onnx, tokens, lexicons, voice_bin, jieba_dir);

    const std::string test_str = "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。";
    std::string test_str1 = "来听一听, 这个是什么口音? How are you doing? Are you ok? Thank you! 你觉得中英文说得如何呢?";
    const std::string test_str2 =
        "该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。";
    const std::string test_str3 =
        "北京时间5月19日多哈世乒赛，王楚钦势如破竹4-0剃光头，零封巴西小将速胜晋级；男单10号种子邱党鏖战七局爆冷被淘汰，从0-3追到3-3，只是最终还是无功而返，下面看看各场对决的简述。王楚钦延续火热的竞技状态，比赛上来连赢七分势不可挡，强力进攻打得对手无可奈何，毫无疑问是做好战术准备，首局几乎没给任何机会11-3速胜。";
    std::string wav_file = "我的祖国.wav";
    kokoro.predict(test_str3, "zf_001", 1, &wav_file);
}
