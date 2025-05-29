//
// Created by aichao on 2025/5/20.
//


#ifdef _WIN32
#include <windows.h>
#endif
#include "capi/audio/tts/kokoro_capi.h"


int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    MDModel model;
    MDKokoroParameters params;
    params.model_path = "../../test_data/test_models/kokoro_v1_1/model.onnx";
    params.tokens_path = "../../test_data/test_models/kokoro_v1_1/tokens.txt";
    params.lexicons_en_path = "../../test_data/test_models/kokoro_v1_1/lexicon-us-en.txt";
    params.lexicons_zh_path = "../../test_data/test_models/kokoro_v1_1/lexicon-zh.txt";
    params.voice_bin_path = "../../test_data/test_models/kokoro_v1_1/voices.bin";
    params.jieba_dir = "../../test_data/test_models/kokoro_v1_1/dict/";
    params.text_normalization_dir = "../../test_data/";
    params.num_threads = -1;
    auto kokoro = md_create_kokoro_model(&model, &params);
    const char* test_str0 = "来听一听, 这个是什么口音? How are you doing your kaldi? Are you ok? Thank you! 你觉得中英文说得如何呢?";
    const char* test_str = "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。";
    const char* test_str1 =
        "来听一听, 这个是什么口音? 现在是北京时间12:24:35，我的电话是18571498072. How are you doing? Are you ok? Thank you! 你觉得中英文说得如何呢?";
    const char* test_str2 =
        "该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。";
    const char* test_str3 =
        "北京时间5月19日多哈世乒赛，王楚钦势如破竹4-0剃光头，零封巴西小将速胜晋级；男单10号种子邱党鏖战七局爆冷被淘汰，从0-3追到3-3，只是最终还是无功而返，下面看看各场对决的简述。王楚钦延续火热的竞技状态，比赛上来连赢七分势不可挡，强力进攻打得对手无可奈何，毫无疑问是做好战术准备，首局几乎没给任何机会11-3速胜。";
    MDTTSResult result;
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
    md_kokoro_model_predict(&model, test_str3, "zf_001", 0.9, &result);
    md_write_wav(&result, "test.wav");
    md_free_kokoro_result(&result);
    md_free_kokoro_model(&model);
}
