//
// capi ASR（SenseVoice）示例
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
    die(md_model_create(&model, MD_MODEL_ASR,
                        "../../test_data/test_models/onnx/sense_voice/model.int8.onnx|"
                        "../../test_data/test_models/onnx/sense_voice/tokens.txt", opt), "create asr");

    const char* text = nullptr;
    die(md_audio_asr_wav(model, "../../test_data/test_models/onnx/sense_voice/test_wavs/zh.wav", &text),
        "asr wav");
    std::printf("recognized: %s\n", text ? text : "(null)");

    md_model_destroy(model);
    md_option_destroy(opt);
    return 0;
}
