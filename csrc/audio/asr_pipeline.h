//
// Created by aichao on 2025/5/19.
//

#pragma once
#include <string>
#include <memory>
#include <deque>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "base_model.h"
#include "audio/asr/sense_voice.h"
#include "audio/asr/paraformer_streaming.h"
#include "audio/vad/silero_vad.h"

namespace modeldeploy::audio {
    // 实时混合 ASR 管线：
    //   - 流式 Paraformer（若配置）边说话边出部分结果 on_asr_partial_；
    //   - VAD 判句结束后，按流式置信度决定：足够高就用流式全文，
    //     否则跑离线 SenseVoice 精修，经 on_asr_ 输出最终结果。
    class MODELDEPLOY_CXX_EXPORT AAsr {
    public:
        AAsr(const std::string& asr_onnx, const std::string& tokens, const std::string& vad_onnx);

        // 带流式 Paraformer 的混合构造（可选第四/五/六参：encoder/decoder/tokens 路径）
        AAsr(const std::string& asr_onnx, const std::string& tokens, const std::string& vad_onnx,
             const std::string& stream_encoder, const std::string& stream_decoder,
             const std::string& stream_tokens, float offline_conf_threshold = 0.55f);

        ~AAsr();
        void push_data(const std::vector<float>& data, int sampleRate);
        void run();
        std::atomic<bool> running_;
        std::thread th_;
        std::unique_ptr<asr::SenseVoice> sense_voice_;
        std::function<void(const std::string& asr)> on_asr_;
        std::function<void(const std::string& partial)> on_asr_partial_;
        void wait_finish();

    private:
        void emit_final();
        std::unique_ptr<asr::ParaformerStreamingAsr> streaming_;
        float conf_threshold_{0.55f};
        std::unique_ptr<vad::SileroVAD> vad_;
        std::deque<float> deque_;
        std::vector<float> cur_wav_;
        std::mutex mutex_;
        std::condition_variable cv_;
    };
}
