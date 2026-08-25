//
// capi TTS（Kokoro）示例
//
#include "../capi_common.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    MDOptionHandle opt = nullptr;
    md_option_create(&opt);
    md_option_set_backend(opt, MD_BK_ORT);
    md_option_set_device(opt, MD_DEV_CPU, 0);
    md_option_set_cpu_threads(opt, 4);

    MDModelHandle model = nullptr;
    die(md_model_create(&model, MD_MODEL_TTS,
                        "../../test_data/test_models/onnx/kokoro_v1_1/model.onnx|"
                        "../../test_data/test_models/onnx/kokoro_v1_1/tokens.txt|"
                        "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-gb-en.txt|"
                        "../../test_data/test_models/onnx/kokoro_v1_1/lexicon-zh.txt|"
                        "../../test_data/test_models/onnx/kokoro_v1_1/voices.bin|"
                        "../../test_data/test_models/onnx/kokoro_v1_1/dict|"
                        "../../test_data", opt), "create tts");

    int sample_rate = 0;
    const float* audio = nullptr;
    size_t n = 0;
    die(md_audio_tts(model, "你好世界今天天气不错 hello world", "zf_001", 1.0f,
                     &sample_rate, &audio, &n), "tts");
    std::printf("synthesized %zu samples @ %d Hz\n", n, sample_rate);
    die(md_wav_save(audio, n, sample_rate, "capi_tts_out.wav"), "save wav");

    md_model_destroy(model);
    md_option_destroy(opt);
    std::puts("OK -> capi_tts_out.wav");
    return 0;
}
