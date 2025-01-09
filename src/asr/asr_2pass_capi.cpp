//
// Created by AC on 2025-01-09.
//
#include <fstream>
#include <sstream>
#include <filesystem>
#include "asr_capi.h"
#include "internal/funasrruntime.h"
#include "internal/com-define.h"
#include "internal/audio.h"
#include "src/utils/internal/utils.h"
#include "asr_2pass_capi.h"


namespace fs = std::filesystem;


struct TwoPassModel {
    void *tpass_handle;
    void *tpass_online_handle;
    void *decoder_model;
    std::vector<std::vector<float>> hot_words_embedding;
    int asr_mode;
};


MDStatusCode md_create_two_pass_model(MDModel *model,
                                      const char *offline_model_dir,
                                      const char *online_model_dir,
                                      const char *vad_dir,
                                      const char *punct_dir,
                                      const char *itn_dir,
                                      const char *lm_dir,
                                      const char *hot_word_path,
                                      ASRMode asr_mode,
                                      float global_beam,
                                      float lattice_beam,
                                      float am_scale,
                                      int fst_inc_wts,
                                      int thread_num) {


    std::map<std::string, std::string> model_path;
    model_path.insert({OFFLINE_MODEL_DIR, offline_model_dir});
    model_path.insert({ONLINE_MODEL_DIR, online_model_dir});
    model_path.insert({QUANTIZE, is_quantize_model(offline_model_dir) ? "true" : "false"});
    model_path.insert({VAD_DIR, vad_dir});
    model_path.insert({VAD_QUANT, is_quantize_model(vad_dir) ? "true" : "false"});
    model_path.insert({PUNC_DIR, punct_dir});
    model_path.insert({PUNC_QUANT, is_quantize_model(vad_dir) ? "true" : "false"});
    model_path.insert({ITN_DIR, itn_dir});
    model_path.insert({LM_DIR, lm_dir});


    void *tpass_handle = FunTpassInit(model_path, thread_num);
    if (!tpass_handle) {
        return MDStatusCode::ModelInitializeFailed;
    }
    float glob_beam = 3.0f;
    float lat_beam = 3.0f;
    float am_sc = 10.0f;
    if (lm_dir) {
        glob_beam = global_beam;
        lat_beam = lattice_beam;
        am_sc = am_scale;
    }
    // init wfst decoder
    void *decoder_handle = FunASRWfstDecoderInit(tpass_handle, ASR_TWO_PASS, glob_beam, lat_beam, am_sc);
    // hotword file
    std::unordered_map<std::string, int> hws_map;
    std::string nn_hot_words;
    ExtractHotWords(hot_word_path, hws_map, nn_hot_words);
    // load hotwords list and build graph
    FunWfstDecoderLoadHwsRes(decoder_handle, fst_inc_wts, hws_map);

    // init online features
    std::vector<int> chunk_size = {5, 10, 5};
    FUNASR_HANDLE tpass_online_handle = FunTpassOnlineInit(tpass_handle, chunk_size);


    std::vector<std::vector<float>> hot_words_embedding = CompileHotwordEmbedding(tpass_handle, nn_hot_words);
    auto model_content = new TwoPassModel{tpass_handle, tpass_online_handle, decoder_handle, hot_words_embedding,
                                          asr_mode};
    model->type = MDModelType::ASR;
    model->format = MDModelFormat::ONNX;
    model->model_content = model_content;
    model->model_name = strdup("FunASR");
    return MDStatusCode::Success;
}

MDStatusCode md_two_pass_model_predict(MDModel *model, const char *wav_path, MDASRResult *asr_result, int audio_fs) {
    funasr::Audio audio(1);
    int32_t sampling_rate = audio_fs;
    std::string wav_file_extension = fs::path(wav_path).extension().string();
    if (wav_file_extension == ".wav") {
        if (!audio.LoadWav2Char(wav_path, &sampling_rate)) {
            std::cerr << "Failed to load " << wav_path << std::endl;
            return MDStatusCode::FileOpenFailed;
        }
    } else if (wav_file_extension == ".pcm") {
        if (!audio.LoadPcmwav2Char(wav_path, &sampling_rate)) {
            std::cerr << "Failed to load " << wav_path << std::endl;
            return MDStatusCode::FileOpenFailed;
        }
    } else {
        if (!audio.FfmpegLoad(wav_path, true)) {
            std::cerr << "Failed to load " << wav_path << std::endl;
            return MDStatusCode::FileOpenFailed;
        }
    }

    char *speech_buff = audio.GetSpeechChar();
    int buff_len = audio.GetSpeechLen() * 2;
    int step = 800 * 2;
    bool is_final = false;
    string online_res;
    string tpass_res;
    string time_stamp_res;
    std::vector<std::vector<string>> punc_cache(2);

    auto model_content = (TwoPassModel *) model->model_content;
    auto asr_mode = (ASRMode) model_content->asr_mode;

    for (int sample_offset = 0; sample_offset < buff_len; sample_offset += std::min(step, buff_len - sample_offset)) {
        if (sample_offset + step >= buff_len - 1) {
            step = buff_len - sample_offset;
            is_final = true;
        } else {
            is_final = false;
        }

        FUNASR_RESULT result = FunTpassInferBuffer(model_content->tpass_handle, model_content->tpass_online_handle,
                                                   speech_buff + sample_offset, step, punc_cache, is_final,
                                                   sampling_rate, "pcm", (ASR_TYPE) asr_mode,
                                                   model_content->hot_words_embedding, true,
                                                   model_content->decoder_model);
        if (result) {
            string online_msg = FunASRGetResult(result, 0);
            online_res += online_msg;
            if (!online_msg.empty()) {
                std::cout << online_msg << std::endl;
            }
            string tpass_msg = FunASRGetTpassResult(result, 0);
            tpass_res += tpass_msg;
            if (!tpass_msg.empty()) {
                std::cout << " offline results : " << tpass_msg << std::endl;
            }
            string stamp = FunASRGetStamp(result);
            if (!stamp.empty()) {
                std::cout << " time stamp : " << stamp << std::endl;
                if (time_stamp_res.empty()) {
                    time_stamp_res += stamp;
                } else {
                    time_stamp_res = time_stamp_res.erase(time_stamp_res.length() - 1)
                                     + "," + stamp.substr(1);
                }
            }
            FunASRFreeResult(result);
        }
    }
    if (asr_mode == ASRMode::TwoPass) {
        LOG(INFO) << " Final online  results " << " : " << online_res;
    }
    if (asr_mode == ASRMode::Online) {
        LOG(INFO) << " Final online  results " << " : " << tpass_res;
    }
    if (asr_mode == ASRMode::Offline || asr_mode == ASRMode::TwoPass) {
        LOG(INFO) << " Final offline results " << " : " << tpass_res;
        if (!time_stamp_res.empty()) {
            LOG(INFO) << " Final timestamp results " << " : " << time_stamp_res;
        }
    }
    return MDStatusCode::Success;
}


void md_free_two_pass_result(MDASRResult *asr_result) {
    if (asr_result != nullptr) {
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
}


void md_free_two_pass_model(MDModel *model) {
    if (model != nullptr) {
        if (model->model_content != nullptr) {
            auto model_content = static_cast<TwoPassModel *> (model->model_content);
            FunWfstDecoderUnloadHwsRes(model_content->decoder_model);
            FunASRWfstDecoderUninit(model_content->decoder_model);
            FunTpassOnlineUninit(model_content->tpass_online_handle);
            FunTpassUninit(model_content->tpass_handle);
            delete model_content;
            model->model_content = nullptr;
        }
        if (model->model_name != nullptr) {
            free(model->model_name);
            model->model_name = nullptr;
        }
    }
}



