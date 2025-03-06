//
// Created by aichao on 2025/2/26.
//

#include "capi/audio/asr/sense_voice_capi.h"
#include "csrc/audio/sherpa-onnx/csrc/c-api.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"


MDStatusCode md_create_sense_voice_model(MDModel* model, MDSenseVoiceParameters* parameters) {
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
    MD_LOG_INFO("========Loading model ...========");
    const SherpaOnnxOfflineRecognizer* recognizer =
        SherpaOnnxCreateOfflineRecognizer(&recognizer_config);
    if (!recognizer) {
        MD_LOG_ERROR(" Model initial failed, please check your parameters");
        return MDStatusCode::ModelInitializeFailed;
    }
    MD_LOG_INFO("========Loading model done========");
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup("sense_voice");
    model->model_content = const_cast<SherpaOnnxOfflineRecognizer*>(recognizer);
    model->type = MDModelType::ASR;
    return MDStatusCode::Success;
}

MDStatusCode md_sense_voice_model_predict(const MDModel* model, const char* wav_path, MDASRResult* asr_result) {
    if (model->type != MDModelType::ASR) {
        MD_LOG_ERROR("Model type is not ASR");
        return MDStatusCode::ModelTypeError;
    }
    const auto wave = SherpaOnnxReadWave(wav_path);
    if (!wave) {
        MD_LOG_ERROR("Failed to read: {}", wav_path);
        return MDStatusCode::FileOpenFailed;
    }
    MD_LOG_INFO("========Start recognition========");
    const auto begin = std::chrono::steady_clock::now();
    const auto recognizer = static_cast<const SherpaOnnxOfflineRecognizer*>(model->model_content);
    const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(recognizer);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples, wave->num_samples);
    SherpaOnnxDecodeOfflineStream(recognizer, stream);
    const SherpaOnnxOfflineRecognizerResult* result = SherpaOnnxGetOfflineStreamResult(stream);
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
    float rtf = elapsed_seconds / snippet_time;
    MD_LOG_INFO("Text is: {}", asr_result->msg);
    MD_LOG_INFO("Snippet time is: {:.2f}", snippet_time);
    MD_LOG_INFO("Elapsed seconds: {:.2f}", elapsed_seconds);
    MD_LOG_INFO("(Real time factor) RTF = {:.2f} / {:.2f} = {:.2f}", elapsed_seconds,
                snippet_time, rtf);
    return MDStatusCode::Success;
}


void md_free_sense_voice_result(MDASRResult* asr_result) {
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


void md_free_sense_voice_model(MDModel* model) {
    if (model->model_content != nullptr) {
        const auto model_content = static_cast<const SherpaOnnxOfflineRecognizer*>(model->model_content);
        SherpaOnnxDestroyOfflineRecognizer(model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
