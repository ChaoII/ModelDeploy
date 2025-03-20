//
// Created by aichao on 2025/2/26.
//

#include <tabulate/tabulate.hpp>
#include "capi/audio/tts/kokoro_capi.h"
#include "csrc/audio/sherpa-onnx/csrc/c-api.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"


MDStatusCode md_create_kokoro_model(MDModel* model, const MDKokoroParameters* parameters) {
    SherpaOnnxOfflineTtsConfig config{};
    config.model.kokoro.model = parameters->model_path;
    config.model.kokoro.voices = parameters->voices_path;
    config.model.kokoro.tokens = parameters->tokens_path;
    config.model.kokoro.data_dir = parameters->data_dir;
    config.model.kokoro.dict_dir = parameters->dict_dir;
    config.model.kokoro.lexicon = parameters->lexicon;
    config.model.num_threads = parameters->num_threads;
    config.model.debug = parameters->debug;
    std::cout << termcolor::magenta << "========Loading model============" << termcolor::reset << std::endl;
    const SherpaOnnxOfflineTts* kokoro_model = SherpaOnnxCreateOfflineTts(&config);
    if (!kokoro_model) {
        MD_LOG_ERROR(" Model initial failed, please check your parameters");
        return MDStatusCode::ModelInitializeFailed;
    }
    std::cout << termcolor::magenta << "========Loading model done=======" << termcolor::reset << std::endl;
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup("kokoro");
    model->model_content = const_cast<SherpaOnnxOfflineTts*>(kokoro_model);
    model->type = MDModelType::TTS;
    return MDStatusCode::Success;
}

MDStatusCode md_kokoro_model_predict(const MDModel* model, const char* text, const int sid, const float speed,
                                     const char* wav_path) {
    if (model->type != MDModelType::TTS) {
        MD_LOG_ERROR("Model type is not TTS");
        return MDStatusCode::ModelTypeError;
    }
    std::cout << termcolor::magenta << "========Start Generating========" << termcolor::reset << std::endl;
    const auto begin = std::chrono::steady_clock::now();
    const auto tts = static_cast<const SherpaOnnxOfflineTts*>(model->model_content);
    const SherpaOnnxGeneratedAudio* audio = SherpaOnnxOfflineTtsGenerate(tts, text, sid, speed);
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, wav_path);
    const auto end = std::chrono::steady_clock::now();
    std::cout << termcolor::magenta << "========Generating finish=========" << termcolor::reset << std::endl;;
    const float elapsed_seconds =
        static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) / 1000.f;
    std::cout << termcolor::cyan << "Generating time: " << termcolor::green
        << elapsed_seconds << " seconds" << termcolor::reset << std::endl;
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    return MDStatusCode::Success;
}

void md_free_kokoro_model(MDModel* model) {
    if (model->model_content != nullptr) {
        const auto model_content = static_cast<const SherpaOnnxOfflineTts*>(model->model_content);
        SherpaOnnxDestroyOfflineTts(model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
