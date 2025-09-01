//
// Created by aichao on 2025/5/20.
//


#include <chrono>  // NOLINT
#include <iostream>
#include <string>
#include "csrc/audio/tts/utils.h"
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
    std::wcout.imbue(std::locale(""));
#endif
    const std::string kokoro_onnx = "../../test_data/test_models/kokoro_v1_1/model.onnx";
    const std::string tokens = "../../test_data/test_models/kokoro_v1_1/tokens.txt";
    const std::vector<std::string> lexicons = {
        "../../test_data/test_models/kokoro_v1_1/lexicon-us-en.txt",
        "../../test_data/test_models/kokoro_v1_1/lexicon-zh.txt"
    };
    const std::string voice_bin = "../../test_data/test_models/kokoro_v1_1/voices.bin";
    const std::string jieba_dir = "../../test_data/test_models/kokoro_v1_1/dict/";
    const std::string text_normalization_dir = "../../test_data/";
    modeldeploy::RuntimeOption option;
    option.use_gpu(0);
    auto kokoro = modeldeploy::audio::tts::Kokoro(kokoro_onnx,
                                                  tokens,
                                                  lexicons, voice_bin,
                                                  jieba_dir,
                                                  text_normalization_dir,
                                                  option);

    const std::string test_str = "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。";
    const std::string test_str0 = "是的，我知道文心一言。文心一言是百度公司推出的一款大型语言模型，和我一样，它也可以回答问题、创作文字、编程、推理、多语言交流等。不过，我们是不同的产品，分别由不同的公司开发。如果你有任何问题，我会尽力帮助你解答。";
    std::string test_str1 =
        "来听一听, 这个是什么口音? 现在是北京时间12:24:35，我的电话是18571498072. How are you doing? Are you ok? Thank you! 你觉得中英文说得如何呢?";
    const std::string test_str2 =
        "该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。";
    const std::string test_str3 =
        "北京时间5月19日多哈世乒赛，王楚钦势如破竹4-0剃光头，零封巴西小将速胜晋级；男单10号种子邱党鏖战七局爆冷被淘汰，从0-3追到3-3，只是最终还是无功而返，下面看看各场对决的简述。王楚钦延续火热的竞技状态，比赛上来连赢七分势不可挡，强力进攻打得对手无可奈何，毫无疑问是做好战术准备，首局几乎没给任何机会11-3速胜。";
    std::vector<float> out_data;
    // voice
    // af_maple,af_sol,bf_vale,zf_001,zf_002,zf_003,zf_004,zf_005,
    // zf_006,zf_007,zf_008,zf_017,zf_018,zf_019,zf_021,zf_022,zf_023,
    // zf_024,zf_026,zf_027,zf_028,zf_032,zf_036,zf_038,zf_039,zf_040,
    // zf_042,zf_043,zf_044,zf_046,zf_047,zf_048,zf_049,zf_051,zf_059,
    // zf_060,zf_067,zf_070,zf_071,zf_072,zf_073,zf_074,zf_075,zf_076,
    // zf_077,zf_078,zf_079,zf_083,zf_084,zf_085,zf_086,zf_087,zf_088,
    // zf_090,zf_092,zf_093,zf_094,zf_099,zm_009,zm_010,zm_011,zm_012,
    // zm_013,zm_014,zm_015,zm_016,zm_020,zm_025,zm_029,zm_030,zm_031,
    // zm_033,zm_034,zm_035,zm_037,zm_041,zm_045,zm_050,zm_052,zm_053,
    // zm_054,zm_055,zm_056,zm_057,zm_058,zm_061,zm_062,zm_063,zm_064,
    // zm_065,zm_066,zm_068,zm_069,zm_080,zm_081,zm_082,zm_089,zm_091,
    // zm_095,zm_096,zm_097,zm_098,zm_100
    kokoro.predict(test_str0, "zf_001", 0.9, &out_data);
    modeldeploy::audio::tts::write_wave(std::string("out.wav"), kokoro.get_sample_rate(), out_data.data(), out_data.size());
}
