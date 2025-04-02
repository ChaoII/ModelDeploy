//
// Created by aichao on 2025/2/26.
//

#include <chrono>
#include "capi/audio/asr/sense_voice_capi.h"
#include "csrc/audio/sherpa-onnx/csrc/c-api.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "tabulate/tabulate.hpp"


MDStatusCode md_create_sense_voice_model(MDModel *model, MDSenseVoiceParameters *parameters) {
    SherpaOnnxOfflineSenseVoiceModelConfig sense_voice_config{};
    sense_voice_config.model = parameters->model_path;
    sense_voice_config.language = parameters->language;
    sense_voice_config.use_itn = parameters->use_itn;
    // Offline model config
    SherpaOnnxOfflineModelConfig offline_model_config{};
    offline_model_config.debug = parameters->debug;
    offline_model_config.num_threads = parameters->num_threads;
    offline_model_config.provider = "cpu";
    offline_model_config.tokens = parameters->tokens_path;
    offline_model_config.sense_voice = sense_voice_config;
    // Recognizer config
    SherpaOnnxOfflineRecognizerConfig recognizer_config{};
    // 贪心搜索
    recognizer_config.decoding_method = "greedy_search";
    recognizer_config.model_config = offline_model_config;
    std::cout << termcolor::magenta << "========Loading model============" << termcolor::reset << std::endl;
    const SherpaOnnxOfflineRecognizer *recognizer =
            SherpaOnnxCreateOfflineRecognizer(&recognizer_config);
    if (!recognizer) {
        MD_LOG_ERROR << " Model initial failed, please check your parameters." << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    std::cout << termcolor::magenta << "========Loading model done=======" << termcolor::reset << std::endl;
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup("sense_voice");
    model->model_content = const_cast<SherpaOnnxOfflineRecognizer *>(recognizer);
    model->type = MDModelType::ASR;
    return MDStatusCode::Success;
}

MDStatusCode md_sense_voice_model_predict(const MDModel *model, const char *wav_path, MDASRResult *asr_result) {
    if (model->type != MDModelType::ASR) {
        MD_LOG_ERROR << "Model type is not ASR." << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto wave = SherpaOnnxReadWave(wav_path);
    if (!wave) {
        MD_LOG_ERROR << "Failed to read: " << wav_path << "." << std::endl;
        return MDStatusCode::FileOpenFailed;
    }
    std::cout << termcolor::magenta << "========Start recognition========" << termcolor::reset << std::endl;
    const auto begin = std::chrono::steady_clock::now();
    const auto recognizer = static_cast<const SherpaOnnxOfflineRecognizer *>(model->model_content);
    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(recognizer);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples, wave->num_samples);
    SherpaOnnxDecodeOfflineStream(recognizer, stream);
    const SherpaOnnxOfflineRecognizerResult *result = SherpaOnnxGetOfflineStreamResult(stream);
    float snippet_time = static_cast<float>(wave->num_samples) / static_cast<float>(wave->sample_rate);
    asr_result->msg = strdup(result->text);
    asr_result->snippet_time = snippet_time;
    asr_result->stamp = strdup("");
    asr_result->stamp_sents = strdup(result->json);
    asr_result->tpass_msg = strdup("");
    const auto end = std::chrono::steady_clock::now();
    const float elapsed_seconds =
            static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000.f;
    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxFreeWave(wave);
    const float rtf = elapsed_seconds / snippet_time;
    std::cout << termcolor::cyan << "Text is: " << termcolor::reset << std::endl;
    std::cout << termcolor::green << asr_result->msg << std::endl;
    std::cout << termcolor::cyan << "Snippet time is: " << termcolor::green << snippet_time << std::endl;
    std::cout << termcolor::cyan << "Elapsed seconds is: " << termcolor::green << elapsed_seconds << std::endl;
    std::cout << termcolor::cyan << "(Real time factor) RTF = " << termcolor::green << elapsed_seconds << "/"
              << snippet_time << " = " << rtf << termcolor::reset << std::endl;
    return MDStatusCode::Success;
}


void md_free_sense_voice_result(MDASRResult *asr_result) {
    if (asr_result->msg != nullptr) {
        free(asr_result->msg);
        asr_result->msg = nullptr;
    }
    if (asr_result->stamp != nullptr) {
        free(asr_result->stamp);
        asr_result->stamp = nullptr;
    }
    if (asr_result->stamp_sents != nullptr) {
        free(asr_result->stamp_sents);
        asr_result->stamp_sents = nullptr;
    }
    if (asr_result->tpass_msg != nullptr) {
        free(asr_result->tpass_msg);
        asr_result->tpass_msg = nullptr;
    }
}


void md_free_sense_voice_model(MDModel *model) {
    if (model->model_content != nullptr) {
        const auto model_content = static_cast<const SherpaOnnxOfflineRecognizer *>(model->model_content);
        SherpaOnnxDestroyOfflineRecognizer(model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
