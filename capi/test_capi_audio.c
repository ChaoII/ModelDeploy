/* capi 音频验证：ASR（SenseVoice）+ TTS（Kokoro）+ wav 落盘 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "md_capi.h"

static int fails = 0;

static void check(const char* name, MDStatus st) {
    if (st != MD_OK) {
        printf("  [FAIL] %s: %s\n", name, md_get_last_error());
        fails++;
    } else {
        printf("  [ok]   %s\n", name);
    }
}

static void run_asr(void) {
    printf("== ASR (SenseVoice) ==\n");
    const char* dir = "test_data/test_models/onnx/sense_voice/";
    char path[1024];
    snprintf(path, sizeof(path), "%smodel.int8.onnx|%stokens.txt", dir, dir);

    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_ASR, path, opt));
    if (m) {
        const char* text = NULL;
        check("predict_wav", md_audio_asr_wav(m, "test_data/test_models/onnx/sense_voice/test_wavs/zh.wav", &text));
        printf("  recognized: '%s'\n", text ? text : "(null)");
        /* 用识别出的文本合成测试 wav 落盘（生成 0.2s 正弦波验证 wav 写出） */
        float tone[4800];
        for (int i = 0; i < 4800; ++i) tone[i] = 0.1f * (float)((i * 440) % 22050) / 22050.0f;
        check("wav_save", md_wav_save(tone, 4800, 24000, "capi/_tmp_tone.wav"));
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

static void run_tts(void) {
    printf("== TTS (Kokoro) ==\n");
    const char* dir = "test_data/test_models/onnx/kokoro_v1_1/";
    char path[2048];
    snprintf(path, sizeof(path), "%smodel.onnx|%stokens.txt|%slexicon-gb-en.txt|%slexicon-zh.txt|%svoices.bin|%sdict|%s",
             dir, dir, dir, dir, dir, dir, dir);

    MDOptionHandle opt = NULL;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle m = NULL;
    check("create", md_model_create(&m, MD_MODEL_TTS, path, opt));
    if (m) {
        /* Kokoro 依赖 text_normalization 目录的 s2t_map.bin/t2s_map.bin；
           测试数据未携带，predict 阶段会因缺文件崩溃，此处仅验证模型加载。 */
        printf("  model loaded OK (predict needs s2t_map.bin/t2s_map.bin data)\n");
        md_model_destroy(m);
    }
    md_option_destroy(opt);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    run_asr();
    run_tts();
    printf("\n%s: %d failure(s)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
