//
// Created by AC on 2025-01-13.
//
#ifdef _WIN32
#include <Windows.h>
#endif


#include <iostream>
#include <chrono>
#include <src/log.h>

#include "src/asr/asr_offline_capi.h"

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    //    MDModel model{strdup(""), {}, {}, nullptr};
    MDModel model{};
    auto start_time1 = std::chrono::steady_clock::now();
    md_create_asr_offline_model(&model,
                                "D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                                "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
                                "D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
                                "D:/funasr-runtime-resources/models/fst_itn_zh",
                                "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst");

    auto end_time1 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1).count();
    MD_LOG_INFO("md_create_asr_offline_model time: {}", duration);

    MDASRResult result{};
    auto start_time2 = std::chrono::steady_clock::now();
    md_asr_offline_model_predict(&model, "D:/funasr-runtime-resources/vad_example.wav", &result);
    auto end_time2 = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2).count();
    MD_LOG_INFO("md_asr_offline_model_predict time: {}", duration);

    MD_LOG_INFO("msg: {}", result.msg);
    MD_LOG_INFO("snippet_time: {}", result.snippet_time);
    MD_LOG_INFO("stamp: {}", result.stamp);
    MD_LOG_INFO("stamp_sents: {}", result.stamp_sents);
    MD_LOG_INFO("tpass_msg: {}", result.tpass_msg);
    MD_LOG_INFO("accelerate rate: {:.1f}x", result.snippet_time*1000/ duration);

    auto start_time3 = std::chrono::steady_clock::now();
    md_free_asr_result(&result);
    md_free_asr_offline_model(&model);
    auto end_time3 = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3).count();
    MD_LOG_INFO("release cost: {}ms", duration);
}
