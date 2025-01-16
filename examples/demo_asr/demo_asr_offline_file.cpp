//
// Created by AC on 2025-01-13.
//
#ifdef _WIN32
#include <Windows.h>
#endif


#include <iostream>
#include "src/asr/asr_offline_capi.h"

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    //    MDModel model{strdup(""), {}, {}, nullptr};
    MDModel model{};
    md_create_asr_offline_model(&model,
                                "D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                                "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                                "D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
                                "D:/funasr-runtime-resources/models/fst_itn_zh",
                                "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst");
    MDASRResult result{};
    md_asr_offline_model_predict(&model, "D:/funasr-runtime-resources/vad_example.wav", &result);


    if (result.msg) {
        std::cout << "----" << result.msg << std::endl;
    }
    if (result.tpass_msg) {
        std::cout << "*****" << result.tpass_msg << std::endl;
    }
    if (result.stamp) {
        std::cout << "++++" << result.stamp << std::endl;
    }
    if (result.stamp_sents) {
        std::cout << "#####" << result.stamp_sents << std::endl;
    }


    md_free_asr_result(&result);

    md_free_asr_offline_model(&model);
}
