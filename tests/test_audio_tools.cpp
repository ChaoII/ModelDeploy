#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "audio/tools/wav_io.h"
#include "audio/tools/audio_meta.h"
#include "audio/tools/itn.h"
#include "audio/tools/hotword.h"
#include "audio/tools/itn_engine.h"
#include "audio/text_normalize/text_normalization.h"
#include "tests/utils.h"
using namespace modeldeploy::audio::tool;

TEST_CASE("ITN normalizes Chinese spoken numbers", "[audio_tools]") {
    InverseTextNormalizer itn;
    REQUIRE(itn.normalize("一百二十三") == "123");
    REQUIRE(itn.normalize("三千零四") == "3004");
    REQUIRE(itn.normalize("十二") == "12");
    REQUIRE(itn.normalize("二十五") == "25");
    REQUIRE(itn.normalize("一百零三万") == "103万");
    REQUIRE(itn.normalize("两千万") == "2000万");
    REQUIRE(itn.normalize("十二亿七千二百万") == "12亿7200万");
    REQUIRE(itn.normalize("三点一四") == "3.14");
    REQUIRE(itn.normalize("百分之五") == "5%");
    REQUIRE(itn.normalize("二零二四年五月九日") == "2024年5月9日");
    REQUIRE(itn.normalize("五月九日") == "5月9日");
    REQUIRE(itn.normalize("五点半") == "5:30");
    REQUIRE(itn.normalize("九点零五分") == "9:05");
    REQUIRE(itn.normalize("第八") == "第8");
    // 模糊词保留（单字/不在数字词里）
    REQUIRE(itn.normalize("十几个人") == "十几个人");
}

TEST_CASE("ITN fraction, currency, measure, telephone", "[audio_tools]") {
    InverseTextNormalizer itn;
    // 分数 X分之Y -> Y/X
    REQUIRE(itn.normalize("四分之三") == "3/4");
    REQUIRE(itn.normalize("三分之二") == "2/3");
    REQUIRE(itn.normalize("十分之一") == "1/10");
    REQUIRE(itn.normalize("五分之一百") == "100/5");
    // 货币
    REQUIRE(itn.normalize("五元六角七分") == "5.67元");
    REQUIRE(itn.normalize("五元六角") == "5.6元");
    REQUIRE(itn.normalize("五十元") == "50元");
    REQUIRE(itn.normalize("五十万元") == "500000元");
    REQUIRE(itn.normalize("三美元") == "3美元");
    REQUIRE(itn.normalize("两千英镑") == "2000英镑");
    // 度量
    REQUIRE(itn.normalize("三公里") == "3公里");
    REQUIRE(itn.normalize("五百克") == "500克");
    REQUIRE(itn.normalize("三十千克") == "30千克");
    REQUIRE(itn.normalize("两吨") == "2吨");
    REQUIRE(itn.normalize("三万公里") == "30000公里");
    // 电话 / 连续号码（含 幺）
    REQUIRE(itn.normalize("幺三八零零一三") == "1380013");
    REQUIRE(itn.normalize("一三八零零一二三四五") == "1380012345");
    // 大写人民币数词归一
    REQUIRE(itn.normalize("壹佰贰拾叁") == "123");
    REQUIRE(itn.normalize("人民币伍拾元") == "人民币50元");
}

TEST_CASE("ItnEngine routes to lightweight backend by default", "[audio_tools]") {
    ItnEngine eng;
    REQUIRE(eng.backend() == ItnBackend::Lightweight);
    REQUIRE(eng.normalize("一百二十三") == "123");
    REQUIRE(eng.normalize("九点零五分") == "9:05");
}

TEST_CASE("ItnEngine WeText backend runs live wetext::Processor pipeline (env-gated)", "[audio_tools]") {
    // 仅当提供了 WeText 语法模型目录时运行（无模型默认 SKIP，保持 ctest 零失败）。
    // 模型目录要求: zh_itn_tagger.fst + zh_itn_verbalizer.fst（wenet 软件栈在 Linux 用 pynini 编译），
    // 或用本地微 FST 验证 pipeline 全链路（构造/Read/Compose/ShortestPath/StringPrinter 已逐字节走通）。
    // 注意：真实 zh_itn tagger 会产出 <...> 标注符；微 identity 模型无标注，Normalize 触发
    // weText 的 TokenParser 异常，SDK 应收敛为 fail-soft（退化轻量）而非外泄异常。
    const char* dir = std::getenv("MODELDEPLOY_WETEXT_DIR");
    if (!dir || !*dir) return;
    ItnEngine eng(ItnBackend::WeText);
    REQUIRE(eng.backend() == ItnBackend::WeText);
    const std::string in = "一百二十三";
    const std::string out = eng.normalize(in);   // 不得抛出异常
    REQUIRE_FALSE(out.empty());
    // 微 identity 模型下 weText Verbalize 失败 -> fail-soft 到轻量: 轻量给出 "123"
    REQUIRE(out == "123");
}

TEST_CASE("HotwordContext scan matches CJK substring and ASCII word boundaries", "[audio_tools]") {
    HotwordContext ctx;
    ctx.add("导航", 2.0f);
    ctx.add("the", 1.0f);
    auto found = ctx.scan("打开导航去the north");
    REQUIRE(found.size() == 2);
    auto it = std::find_if(found.begin(), found.end(), [](const FoundHotword& h){ return h.word == "导航"; });
    REQUIRE(it != found.end());
    REQUIRE(it->count == 1);
    REQUIRE(it->weight == Catch::Approx(2.0f));
}

TEST_CASE("HotwordContext score via ContextGraph boosts matched text", "[audio_tools]") {
    HotwordContext ctx;
    ctx.add("会议", 3.0f);
    auto g = build_context_graph(ctx, tokenize_chars);
    REQUIRE(score(*g, tokenize_chars, "会议室") > score(*g, tokenize_chars, "普通场所"));
    HotwordContext empty;
    auto g0 = build_context_graph(empty, tokenize_chars);
    REQUIRE(score(*g0, tokenize_chars, "会议室") == Catch::Approx(0.0f));
}

TEST_CASE("HotwordContext rescore reorders hypotheses by context biasing", "[audio_tools]") {
    HotwordContext ctx;
    ctx.add("天气", 5.0f);
    std::vector<Hypothesis> hyps = {
        {"今天很热", 10.0f},
        {"今天天气不错", 8.0f},
        {"天气", 6.0f},
    };
    auto ranked = rescore(std::move(hyps), ctx, tokenize_chars, 1.0f);
    REQUIRE(ranked.size() == 3);
    REQUIRE(ranked[0].text == "今天天气不错"); // 8 + boost
    REQUIRE(ranked[1].text == "天气");         // 6 + boost
    REQUIRE(ranked[2].text == "今天很热");     // 10 + 0
}

TEST_CASE("HotwordContext highlight wraps matched hotwords", "[audio_tools]") {
    HotwordContext ctx;
    ctx.add("天窗", 1.0f);
    ctx.add("天气", 1.0f);
    REQUIRE(ctx.highlight("今天天气晴") == "今天[天气]晴");
}

TEST_CASE("WavIO write then read roundtrip", "[audio_tools]") {
    std::vector<float> sine(1600);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    const std::string path = "audio_roundtrip_test.wav";
    REQUIRE(write_wav(path, sine, 16000));
    WavData d;
    REQUIRE(read_wav(path, &d));
    REQUIRE(d.meta.sample_rate == 16000);
    REQUIRE(d.meta.channels == 1);
    REQUIRE(d.samples.size() == sine.size());
    REQUIRE(d.samples[100] == Catch::Approx(sine[100]).margin(1e-3f));  // 16-bit 量化误差
    auto meta = parse_meta(path);
    REQUIRE(meta.sample_rate == 16000);
    REQUIRE(meta.bits == 16);
    std::remove(path.c_str());
}
#include "audio/tools/resampler.h"

TEST_CASE("Resampler 8k->16k doubles length, frequency preserved", "[audio_tools]") {
    std::vector<float> sine(800);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = std::sin(2 * 3.14159265f * 1000.0f * (i / 8000.0f));
    auto out = Resampler::resample(sine, 8000, 16000);
    REQUIRE(out.size() == sine.size() * 2);
    const float* p = out.data();
    const size_t N = out.size();
    const double target = (double)N * 1000.0 / 16000.0;
    double best1 = 0.0, best2 = 0.0; size_t bestb = 0;
    for (size_t k = 0; k < N / 2; ++k) {
        double re = 0, im = 0;
        for (size_t i = 0; i < N; ++i) {
            const double a = 2 * 3.14159265 * k * i / N;
            re += p[i] * std::cos(a); im -= p[i] * std::sin(a);
        }
        const double e = re * re + im * im;
        if (e > best2) { best2 = best1; best1 = e; bestb = k; }
        (void)best2;
    }
    REQUIRE(std::abs((double)(int)bestb - target) <= 2);
}
#include "audio/tools/fbank.h"

TEST_CASE("Fbank produces frames x bins non-degenerate", "[audio_tools]") {
    Fbank fb(16000, 80);
    std::vector<float> s(16000);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    auto frames = fb.compute(s);
    REQUIRE_FALSE(frames.empty());
    REQUIRE(frames[0].size() == 80);
    float energy = 0;
    for (const auto& row : frames) for (float v : row) energy += v * v;
    REQUIRE(energy > 0.0f);
}
#include "audio/tools/waveform.h"

TEST_CASE("Spectrum single tone peak bin", "[audio_tools]") {
    const size_t N = 1024;
    std::vector<float> s(N);
    for (size_t i = 0; i < N; ++i) s[i] = std::sin(2 * 3.14159265f * 100.0f * (i / 1024.0f));
    Spectrum sp(1024);
    auto mag = sp.magnitudes(s);
    REQUIRE(mag.size() == N / 2 + 1);
    size_t best = 0; float maxv = -1;
    for (size_t k = 0; k < mag.size(); ++k) if (mag[k] > maxv) { maxv = mag[k]; best = k; }
    REQUIRE((size_t)(std::abs((long)((int)best - 100))) <= 2);
}

TEST_CASE("Waveform downsample reduces length", "[audio_tools]") {
    std::vector<float> s(4000, 0.5f);
    auto d = Waveform::downsample(s, 200);
    REQUIRE(d.size() <= 200);
}
#include "audio/tools/vad_segment.h"

TEST_CASE("VadSegment splits speech vs silence", "[audio_tools]") {
    const int sr = 16000;
    std::vector<float> sig;
    auto tone = [&](float amp, float dur_s){ for (int i = 0; i < (int)(sr*dur_s); ++i) sig.push_back(amp * std::sin(2*3.14159265f*440.0f*(i/(double)sr))); };
    auto silence = [&](float dur_s){ sig.resize(sig.size() + (size_t)(sr*dur_s), 0.0f); };
    tone(0.9f, 1.0f); silence(0.5f); tone(0.9f, 1.0f);
    VadSegment vad(sr, 0.01f, 200, 200);
    vad.feed(sig);
    auto segs = vad.segments();
    REQUIRE(segs.size() >= 2);
    REQUIRE(segs[0].samples.size() > 0);
    REQUIRE(segs[1].start_ms > segs[0].end_ms);
}

TEST_CASE("TextNormalizer baseline (正向 TN 现状快照)", "[audio_tools]") {
    const std::filesystem::path data_dir = get_test_data_path();
    if (!std::filesystem::exists(data_dir / "s2t_map.bin")) {
        WARN("test_data/s2t_map.bin 缺失，跳过");
        return;
    }
    modeldeploy::audio::TextNormalizer tn(data_dir);
    // 探测型基线：用 CHECK 收集实际输出（首次运行失败信息即真实输出，随后固化为基线）。
    CHECK(tn.normalize_sentence(L"3.14") == L"三点一四");
    CHECK(tn.normalize_sentence(L"123") == L"一百二十三");
    CHECK(tn.normalize_sentence(L"3/4") == L"四分之三");
    CHECK(tn.normalize_sentence(L"60%") == L"百分之六十");
    CHECK(tn.normalize_sentence(L"30度") == L"三十度");
    CHECK(tn.normalize_sentence(L"10m2") == L"十平方米");
    CHECK(tn.normalize_sentence(L"13800123456") == L"幺三八零零幺二三四五六");
    // 现状探测：负数/量词/大数 是否已被通用 re_number 覆盖（决定是否需启用注释规则）
    CHECK(tn.normalize_sentence(L"-5") == L"负五");
    CHECK(tn.normalize_sentence(L"300人") == L"三百人");
    CHECK(tn.normalize_sentence(L"50件") == L"五十件");
    CHECK(tn.normalize_sentence(L"123456") == L"十二万三千四百五十六");
}

TEST_CASE("TextNormalizer measure dict expansion (RED)", "[audio_tools]") {
    const std::filesystem::path data_dir = get_test_data_path();
    if (!std::filesystem::exists(data_dir / "s2t_map.bin")) {
        WARN("test_data/s2t_map.bin 缺失，跳过");
        return;
    }
    modeldeploy::audio::TextNormalizer tn(data_dir);
    // RED：当前 measure_dict 缺这些"符号->中文量词"映射，这些输入会保持原样或错读。
    REQUIRE(tn.normalize_sentence(L"5kg") == L"五千克");
    REQUIRE(tn.normalize_sentence(L"10km") == L"十千米");
    REQUIRE(tn.normalize_sentence(L"2cm") == L"二厘米");
    REQUIRE(tn.normalize_sentence(L"10mm") == L"十毫米");
    REQUIRE(tn.normalize_sentence(L"500ml") == L"五百毫升");
}

TEST_CASE("TextNormalizer 400 uniform number (RED)", "[audio_tools]") {
    const std::filesystem::path data_dir = get_test_data_path();
    if (!std::filesystem::exists(data_dir / "s2t_map.bin")) {
        WARN("test_data/s2t_map.bin 缺失，跳过");
        return;
    }
    modeldeploy::audio::TextNormalizer tn(data_dir);
    // RED：process_uniform_number 硬编码 "400" 且仅替换到连字符号码段，
    // 无连字符的 400 号码无法完整读取。期望按位读出整串。
    REQUIRE(tn.normalize_sentence(L"4001234567") == L"四零零幺二三四五六七");
    REQUIRE(tn.normalize_sentence(L"400-123-4567") == L"四零零幺二三四五六七");
}
