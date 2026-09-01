// csrc/audio/tts/audio8/audio8_runtime.cpp
#include "audio/tts/audio8/audio8_runtime.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <onnxruntime_cxx_api.h>

#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

#include "audio/tts/common/ort_ep.h"
#include "core/md_log.h"

namespace modeldeploy::audio::tts::audio8 {
namespace {

// ORT 的 TypeToTensorType<uint16_t> 映射为 UINT16 而非 FLOAT16，fp16 输入需显式类型。
Ort::Value CreateFP16Tensor(const Ort::MemoryInfo& info, void* data, size_t count,
                            const int64_t* shape, size_t ndim) {
    return Ort::Value::CreateTensor(info, data, count * sizeof(uint16_t), shape, ndim,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}

Ort::Value CreateBoolTensor(const Ort::MemoryInfo& info, void* data, size_t count,
                            const int64_t* shape, size_t ndim) {
    return Ort::Value::CreateTensor(info, data, count, shape, ndim,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

std::string ResolvePath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

}  // namespace

struct Audio8Runtime::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR};
    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> slow;
    std::unique_ptr<Ort::Session> fast;
    std::unique_ptr<Ort::Session> decoder;

    // feed 名（顺序即 Run 时 values 顺序）
    std::vector<std::string> slow_in_names, fast_in_names, dec_in_names;
    std::vector<std::string> slow_out_names, fast_out_names, dec_out_names;
    std::vector<const char*> slow_in_ptrs, fast_in_ptrs, dec_in_ptrs;
    std::vector<const char*> slow_out_ptrs, fast_out_ptrs, dec_out_ptrs;

    // 配置
    int64_t num_layers = 24;
    int64_t num_fast_layers = 4;
    int64_t num_codebooks = 10;
    int64_t n_local_heads = 2;
    int64_t fast_n_local_heads = 2;
    int64_t head_dim = 64;
    int64_t fast_head_dim = 64;
    int64_t max_seq_len = 2048;
    int64_t slow_logits_size = 4097;
    int64_t fast_dim = 896;

    bool loaded = false;

    bool LoadSession(const std::string& path, const char* tag,
                     std::unique_ptr<Ort::Session>* out) {
        if (path.empty() || !std::ifstream(path, std::ios::binary)) {
            MD_LOG_ERROR << "audio8: cannot open onnx " << tag << ": " << path << std::endl;
            return false;
        }
        try {
            // ORT 在 Windows 下路径 API 为 wchar_t（ORTCHAR_T）；std::filesystem::path
            // 在 Windows 上即 wchar_t 值类型，可直接满足。路径构造可正确处理外部权重
            // （.onnx.data 与 onnx 同目录）。
#ifdef _WIN32
            *out = std::make_unique<Ort::Session>(env, std::filesystem::path(path).c_str(),
                                                  opts);
#else
            *out = std::make_unique<Ort::Session>(env, path.c_str(), opts);
#endif
        } catch (const Ort::Exception& e) {
            MD_LOG_ERROR << "audio8: failed to load " << tag << " (" << path
                         << "): " << e.what() << std::endl;
            return false;
        }
        MD_LOG_INFO << "audio8: loaded " << tag << " session: " << path << std::endl;
        return true;
    }
};

Audio8Runtime::Audio8Runtime() : impl_(std::make_unique<Impl>()) {}

Audio8Runtime::~Audio8Runtime() = default;

bool Audio8Runtime::Load(const Audio8Manifest& manifest, int32_t threads, Device device,
                         int32_t device_id) {
    if (!impl_) return false;
    Impl& I = *impl_;
    I.num_layers = manifest.num_layers;
    I.num_fast_layers = manifest.num_fast_layers;
    I.num_codebooks = manifest.num_codebooks;
    I.n_local_heads = manifest.n_local_heads;
    I.fast_n_local_heads = manifest.fast_n_local_heads;
    I.head_dim = manifest.head_dim;
    I.fast_head_dim = manifest.fast_head_dim;
    I.max_seq_len = manifest.max_seq_len;
    I.slow_logits_size = manifest.slow_logits_size;
    I.fast_dim = manifest.fast_dim;

    num_codebooks_ = manifest.num_codebooks;
    slow_logits_size_ = manifest.slow_logits_size;
    fast_dim_ = manifest.fast_dim;

    I.opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    I.opts.SetLogSeverityLevel(3);
    if (threads > 0) I.opts.SetIntraOpNumThreads(threads);
    I.opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    ApplyOrtCudaEp(I.opts, device, device_id);

    const std::string slow_path = ResolvePath(manifest.model_dir, manifest.slow_model);
    const std::string fast_path = ResolvePath(manifest.model_dir, manifest.fast_model);
    const std::string dec_path = ResolvePath(manifest.model_dir, manifest.codec_model);
    if (!I.LoadSession(slow_path, "slow", &I.slow)) return false;
    if (!I.LoadSession(fast_path, "fast", &I.fast)) return false;
    if (!I.LoadSession(dec_path, "codec-decoder", &I.decoder)) return false;

    // ---- slow 输入输出名 ----
    I.slow_in_names.clear();
    I.slow_in_names.push_back("codes");
    I.slow_in_names.push_back("input_pos");
    for (int64_t i = 0; i < I.num_layers; ++i) {
        I.slow_in_names.push_back("cache_key_" + std::to_string(i));
        I.slow_in_names.push_back("cache_value_" + std::to_string(i));
    }
    I.slow_out_names.clear();
    I.slow_out_names.push_back("logits");
    I.slow_out_names.push_back("slow_hidden");
    for (int64_t i = 0; i < I.num_layers; ++i) {
        I.slow_out_names.push_back("key_delta_" + std::to_string(i));
        I.slow_out_names.push_back("value_delta_" + std::to_string(i));
    }
    I.slow_in_ptrs.clear();
    I.slow_in_ptrs.reserve(I.slow_in_names.size());
    for (const auto& n : I.slow_in_names) I.slow_in_ptrs.push_back(n.c_str());
    I.slow_out_ptrs.clear();
    I.slow_out_ptrs.reserve(I.slow_out_names.size());
    for (const auto& n : I.slow_out_names) I.slow_out_ptrs.push_back(n.c_str());

    // ---- fast 输入输出名 ----
    I.fast_in_names.clear();
    I.fast_in_names.push_back("slow_hidden");
    I.fast_in_names.push_back("token_id");
    I.fast_in_names.push_back("use_slow_hidden");
    I.fast_in_names.push_back("input_pos");
    for (int64_t i = 0; i < I.num_fast_layers; ++i) {
        I.fast_in_names.push_back("cache_key_" + std::to_string(i));
        I.fast_in_names.push_back("cache_value_" + std::to_string(i));
    }
    I.fast_out_names.clear();
    I.fast_out_names.push_back("logits");
    for (int64_t i = 0; i < I.num_fast_layers; ++i) {
        I.fast_out_names.push_back("key_delta_" + std::to_string(i));
        I.fast_out_names.push_back("value_delta_" + std::to_string(i));
    }
    I.fast_in_ptrs.clear();
    I.fast_in_ptrs.reserve(I.fast_in_names.size());
    for (const auto& n : I.fast_in_names) I.fast_in_ptrs.push_back(n.c_str());
    I.fast_out_ptrs.clear();
    I.fast_out_ptrs.reserve(I.fast_out_names.size());
    for (const auto& n : I.fast_out_names) I.fast_out_ptrs.push_back(n.c_str());

    // ---- decoder 输入输出名 ----
    I.dec_in_names = {"codes"};
    I.dec_out_names = {"audio"};
    I.dec_in_ptrs = {"codes"};
    I.dec_out_ptrs = {"audio"};

    loaded_ = true;
    MD_LOG_INFO << "audio8: runtime loaded (threads=" << threads << ")" << std::endl;
    return true;
}

bool Audio8Runtime::SlowStep(const std::vector<int64_t>& codes,
                             const std::vector<int64_t>& positions,
                             std::vector<uint16_t>* cache, std::vector<float>* last_logits,
                             std::vector<uint16_t>* last_hidden) {
    if (!impl_ || !impl_->slow || !cache || !last_logits || !last_hidden) return false;
    Impl& I = *impl_;
    const int64_t T = static_cast<int64_t>(positions.size());
    const int64_t rows = num_codebooks_ + 1;
    if (static_cast<int64_t>(codes.size()) != rows * T) {
        MD_LOG_ERROR << "audio8: slow codes size mismatch" << std::endl;
        return false;
    }
    const int64_t seg = I.n_local_heads * I.max_seq_len * I.head_dim;
    if (static_cast<int64_t>(cache->size()) != 2 * I.num_layers * seg) {
        MD_LOG_ERROR << "audio8: slow cache buffer size mismatch" << std::endl;
        return false;
    }

    std::vector<Ort::Value> feeds;
    feeds.reserve(I.slow_in_ptrs.size());
    const std::array<int64_t, 3> codes_shape{1, rows, T};
    feeds.push_back(Ort::Value::CreateTensor<int64_t>(
        I.meminfo, const_cast<int64_t*>(codes.data()), codes.size(), codes_shape.data(), 3));
    const std::array<int64_t, 1> pos_shape{T};
    feeds.push_back(Ort::Value::CreateTensor<int64_t>(
        I.meminfo, const_cast<int64_t*>(positions.data()), positions.size(), pos_shape.data(), 1));
    const std::array<int64_t, 4> cache_shape{1, I.n_local_heads, I.max_seq_len, I.head_dim};
    for (int64_t i = 0; i < I.num_layers; ++i) {
        feeds.push_back(CreateFP16Tensor(I.meminfo, cache->data() + (2 * i) * seg,
                                         static_cast<size_t>(seg), cache_shape.data(), 4));
        feeds.push_back(CreateFP16Tensor(I.meminfo, cache->data() + (2 * i + 1) * seg,
                                         static_cast<size_t>(seg), cache_shape.data(), 4));
    }

    try {
        std::vector<Ort::Value> outs =
            I.slow->Run(Ort::RunOptions{nullptr}, I.slow_in_ptrs.data(), feeds.data(),
                        feeds.size(), I.slow_out_ptrs.data(), I.slow_out_ptrs.size());
        if (outs.size() < 2 + 2 * static_cast<size_t>(I.num_layers)) {
            MD_LOG_ERROR << "audio8: slow run returned unexpected outputs" << std::endl;
            return false;
        }
        // logits [1, 1, slow_logits_size]（导出模型仅输出最后位置的 logits）
        {
            const float* logits = outs[0].GetTensorData<float>();
            const int64_t count = I.slow_logits_size;
            last_logits->assign(logits, logits + count);
        }
        // slow_hidden [1, 1, fast_dim] (fp16 位模式) -> 最后一行
        {
            const uint16_t* hidden = outs[1].GetTensorData<uint16_t>();
            const int64_t count = I.fast_dim;
            last_hidden->assign(hidden, hidden + count);
        }
        // 写回 delta（key_delta_i/ value_delta_i 形状 [1, h, T, d]）
        const int64_t pos0 = positions[0];
        for (int64_t i = 0; i < I.num_layers; ++i) {
            const uint16_t* kd = outs[2 + 2 * i].GetTensorData<uint16_t>();
            const uint16_t* vd = outs[3 + 2 * i].GetTensorData<uint16_t>();
            uint16_t* kbase = cache->data() + (2 * i) * seg;
            uint16_t* vbase = cache->data() + (2 * i + 1) * seg;
            for (int64_t h = 0; h < I.n_local_heads; ++h) {
                for (int64_t t = 0; t < T; ++t) {
                    const int64_t dst = (h * I.max_seq_len + pos0 + t) * I.head_dim;
                    std::memcpy(kbase + dst, kd + (h * T + t) * I.head_dim,
                                static_cast<size_t>(I.head_dim) * sizeof(uint16_t));
                    std::memcpy(vbase + dst, vd + (h * T + t) * I.head_dim,
                                static_cast<size_t>(I.head_dim) * sizeof(uint16_t));
                }
            }
        }
    } catch (const Ort::Exception& e) {
        MD_LOG_ERROR << "audio8: slow run failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool Audio8Runtime::FastStep(int64_t token, bool use_hidden, int64_t position,
                             const std::vector<uint16_t>& slow_hidden,
                             std::vector<uint16_t>* fast_cache,
                             std::vector<float>* last_logits) {
    if (!impl_ || !impl_->fast || !fast_cache || !last_logits) return false;
    Impl& I = *impl_;
    const int64_t seg = I.fast_n_local_heads * I.num_codebooks * I.fast_head_dim;
    if (static_cast<int64_t>(fast_cache->size()) != 2 * I.num_fast_layers * seg) {
        MD_LOG_ERROR << "audio8: fast cache buffer size mismatch" << std::endl;
        return false;
    }
    if (static_cast<int64_t>(slow_hidden.size()) != I.fast_dim) {
        MD_LOG_ERROR << "audio8: slow_hidden size mismatch" << std::endl;
        return false;
    }

    std::vector<Ort::Value> feeds;
    feeds.reserve(I.fast_in_ptrs.size());
    const std::array<int64_t, 3> hidden_shape{1, 1, I.fast_dim};
    feeds.push_back(CreateFP16Tensor(I.meminfo, const_cast<uint16_t*>(slow_hidden.data()),
                                     slow_hidden.size(), hidden_shape.data(), 3));
    const std::array<int64_t, 2> token_shape{1, 1};
    const std::array<int64_t, 1> scalar_shape{1};
    std::array<int64_t, 1> token_val{token};
    feeds.push_back(Ort::Value::CreateTensor<int64_t>(I.meminfo, token_val.data(), token_val.size(),
                                                      token_shape.data(), 2));
    std::array<uint8_t, 1> use_val{static_cast<uint8_t>(use_hidden ? 1 : 0)};
    feeds.push_back(CreateBoolTensor(I.meminfo, use_val.data(), 1, scalar_shape.data(), 1));
    std::array<int64_t, 1> pos_val{position};
    feeds.push_back(Ort::Value::CreateTensor<int64_t>(I.meminfo, pos_val.data(), pos_val.size(),
                                                      scalar_shape.data(), 1));
    const std::array<int64_t, 4> cache_shape{1, I.fast_n_local_heads, I.num_codebooks,
                                             I.fast_head_dim};
    for (int64_t i = 0; i < I.num_fast_layers; ++i) {
        feeds.push_back(CreateFP16Tensor(I.meminfo, fast_cache->data() + (2 * i) * seg,
                                         static_cast<size_t>(seg), cache_shape.data(), 4));
        feeds.push_back(CreateFP16Tensor(I.meminfo, fast_cache->data() + (2 * i + 1) * seg,
                                         static_cast<size_t>(seg), cache_shape.data(), 4));
    }

    try {
        std::vector<Ort::Value> outs =
            I.fast->Run(Ort::RunOptions{nullptr}, I.fast_in_ptrs.data(), feeds.data(),
                        feeds.size(), I.fast_out_ptrs.data(), I.fast_out_ptrs.size());
        if (outs.size() < 1 + 2 * static_cast<size_t>(I.num_fast_layers)) {
            MD_LOG_ERROR << "audio8: fast run returned unexpected outputs" << std::endl;
            return false;
        }
        {
            const float* logits = outs[0].GetTensorData<float>();
            const int64_t count = 4096;  // codebook_size = 4096
            last_logits->assign(logits, logits + count);
        }
        for (int64_t i = 0; i < I.num_fast_layers; ++i) {
            const uint16_t* kd = outs[1 + 2 * i].GetTensorData<uint16_t>();
            const uint16_t* vd = outs[2 + 2 * i].GetTensorData<uint16_t>();
            uint16_t* kbase = fast_cache->data() + (2 * i) * seg;
            uint16_t* vbase = fast_cache->data() + (2 * i + 1) * seg;
            for (int64_t h = 0; h < I.fast_n_local_heads; ++h) {
                const int64_t dst = (h * I.num_codebooks + position) * I.fast_head_dim;
                std::memcpy(kbase + dst, kd + h * I.fast_head_dim,
                            static_cast<size_t>(I.fast_head_dim) * sizeof(uint16_t));
                std::memcpy(vbase + dst, vd + h * I.fast_head_dim,
                            static_cast<size_t>(I.fast_head_dim) * sizeof(uint16_t));
            }
        }
    } catch (const Ort::Exception& e) {
        MD_LOG_ERROR << "audio8: fast run failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool Audio8Runtime::DecodeCodes(const std::vector<int64_t>& codes, int64_t frames,
                                std::vector<float>* audio) {
    if (!impl_ || !impl_->decoder || !audio) return false;
    Impl& I = *impl_;
    if (static_cast<int64_t>(codes.size()) != I.num_codebooks * frames) {
        MD_LOG_ERROR << "audio8: decoder codes size mismatch" << std::endl;
        return false;
    }
    const std::array<int64_t, 3> codes_shape{1, I.num_codebooks, frames};
    std::vector<Ort::Value> feeds;
    feeds.push_back(Ort::Value::CreateTensor<int64_t>(
        I.meminfo, const_cast<int64_t*>(codes.data()), codes.size(), codes_shape.data(), 3));
    try {
        std::vector<Ort::Value> outs =
            I.decoder->Run(Ort::RunOptions{nullptr}, I.dec_in_ptrs.data(), feeds.data(),
                           feeds.size(), I.dec_out_ptrs.data(), I.dec_out_ptrs.size());
        if (outs.empty()) {
            MD_LOG_ERROR << "audio8: decoder run returned no outputs" << std::endl;
            return false;
        }
        const auto info = outs[0].GetTensorTypeAndShapeInfo();
        const size_t total = info.GetElementCount();
        const float* data = outs[0].GetTensorData<float>();
        audio->assign(data, data + total);
    } catch (const Ort::Exception& e) {
        MD_LOG_ERROR << "audio8: decoder run failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

void Audio8Runtime::ReserveSpeech(size_t num_frames) {
    // 预留解码缓冲（StreamWindow 由调用方管理，这里仅提示未来的帧规模）
    (void)num_frames;
}

// Audio8GpuState 结构与其 deleter 定义无条件编译(F1:否则 CPU+BUILD_TESTS 构建链接失败)。
// 结构仅依赖 ORT 类型(Ort::MemoryInfo/Allocator/IoBinding/指针),不含 CUDA 符号;
// CPU 构建不会构造其实例,而测试 TU 析构 Audio8GpuStatePtr 需要 deleter 符号可链接。
struct Audio8Runtime::Audio8GpuState {
    Ort::MemoryInfo cuda_mem{nullptr};
    Ort::Allocator slow_alloc{nullptr};   // 归属:释放缓冲用(ORT arena 本身不 Free 到 CUDA 池)
    Ort::Allocator fast_alloc{nullptr};
    // slow:2*num_layers 个 K/V 段,每段 seg 个 fp16(cache 布局 [h][seq][d])
    std::vector<void*> slow_kv;
    // slow scratch:delta 输出连续区(每段 seg 个 fp16;每次按当帧 [1,h,T,d] 复用前缀)
    std::vector<void*> slow_scratch;
    int64_t slow_seg = 0;
    // fast 同理
    std::vector<void*> fast_kv;
    std::vector<void*> fast_scratch;
    int64_t fast_seg = 0;
    std::unique_ptr<Ort::IoBinding> slow_binding;
    std::unique_ptr<Ort::IoBinding> fast_binding;
};

void Audio8Runtime::Audio8GpuStateDeleter::operator()(Audio8GpuState* p) const noexcept {
    if (!p) return;
    for (auto* buf : p->slow_kv) p->slow_alloc.Free(buf);
    for (auto* buf : p->slow_scratch) p->slow_alloc.Free(buf);
    for (auto* buf : p->fast_kv) p->fast_alloc.Free(buf);
    for (auto* buf : p->fast_scratch) p->fast_alloc.Free(buf);
    delete p;
}

#ifdef WITH_GPU

Audio8Runtime::Audio8GpuStatePtr
Audio8Runtime::MakeGpuState(Device device, int32_t device_id) {
    if (!impl_ || !impl_->slow || !impl_->fast || device != Device::GPU) return {};
    const auto provs = Ort::GetAvailableProviders();
    if (std::find(provs.begin(), provs.end(), "CUDAExecutionProvider") == provs.end()) {
        MD_LOG_WARN << "audio8: CUDA provider unavailable, GPU KV state disabled."
                    << std::endl;
        return {};
    }
    Impl& I = *impl_;
    Audio8Runtime::Audio8GpuStatePtr g(new Audio8GpuState());
    g->cuda_mem = Ort::MemoryInfo("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    g->slow_alloc = Ort::Allocator(*I.slow, g->cuda_mem);
    g->fast_alloc = Ort::Allocator(*I.fast, g->cuda_mem);
    const int64_t seg = I.n_local_heads * I.max_seq_len * I.head_dim;
    g->slow_seg = seg;
    g->fast_seg = I.fast_n_local_heads * I.num_codebooks * I.fast_head_dim;
    try {
        g->slow_kv.resize(static_cast<size_t>(2 * I.num_layers));
        g->slow_scratch.resize(static_cast<size_t>(2 * I.num_layers));
        for (auto& p : g->slow_kv) p = g->slow_alloc.Alloc(static_cast<size_t>(seg) * 2);
        for (auto& p : g->slow_scratch) p = g->slow_alloc.Alloc(static_cast<size_t>(seg) * 2);
        g->fast_kv.resize(static_cast<size_t>(2 * I.num_fast_layers));
        g->fast_scratch.resize(static_cast<size_t>(2 * I.num_fast_layers));
        for (auto& p : g->fast_kv) p = g->fast_alloc.Alloc(static_cast<size_t>(g->fast_seg) * 2);
        for (auto& p : g->fast_scratch) p = g->fast_alloc.Alloc(static_cast<size_t>(g->fast_seg) * 2);
    } catch (const std::exception& e) {
        MD_LOG_WARN << "audio8: GPU KV alloc failed: " << e.what() << std::endl;
        return {};
    }
    // 初始全零(逐段,勿拼连续);失败 fail-closed(F3)
    for (auto* p : g->slow_kv) {
        if (cudaMemset(p, 0, static_cast<size_t>(seg) * 2) != cudaSuccess) {
            MD_LOG_WARN << "audio8: cudaMemset slow_kv failed" << std::endl;
            return {};
        }
    }
    for (auto* p : g->slow_scratch) {
        if (cudaMemset(p, 0, static_cast<size_t>(seg) * 2) != cudaSuccess) {
            MD_LOG_WARN << "audio8: cudaMemset slow_scratch failed" << std::endl;
            return {};
        }
    }
    for (auto* p : g->fast_kv) {
        if (cudaMemset(p, 0, static_cast<size_t>(g->fast_seg) * 2) != cudaSuccess) {
            MD_LOG_WARN << "audio8: cudaMemset fast_kv failed" << std::endl;
            return {};
        }
    }
    for (auto* p : g->fast_scratch) {
        if (cudaMemset(p, 0, static_cast<size_t>(g->fast_seg) * 2) != cudaSuccess) {
            MD_LOG_WARN << "audio8: cudaMemset fast_scratch failed" << std::endl;
            return {};
        }
    }

    // ---- slow IoBinding:cache 输入绑一次;logits/slow_hidden 绑 CPU;delta 每帧 Value-绑 ----
    g->slow_binding = std::make_unique<Ort::IoBinding>(*I.slow);
    {
        const std::array<int64_t, 4> cache_shape{1, I.n_local_heads, I.max_seq_len, I.head_dim};
        for (int64_t i = 0; i < I.num_layers; ++i) {
            auto kv = CreateFP16Tensor(g->cuda_mem,
                                       static_cast<uint16_t*>(g->slow_kv[static_cast<size_t>(2 * i)]),
                                       static_cast<size_t>(seg), cache_shape.data(), 4);
            auto vv = CreateFP16Tensor(g->cuda_mem,
                                       static_cast<uint16_t*>(g->slow_kv[static_cast<size_t>(2 * i + 1)]),
                                       static_cast<size_t>(seg), cache_shape.data(), 4);
            g->slow_binding->BindInput(("cache_key_" + std::to_string(i)).c_str(), kv);
            g->slow_binding->BindInput(("cache_value_" + std::to_string(i)).c_str(), vv);
        }
        const Ort::MemoryInfo cpu_mem =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        g->slow_binding->BindOutput("logits", cpu_mem);
        g->slow_binding->BindOutput("slow_hidden", cpu_mem);
    }
    // ---- fast IoBinding(绑一次)----
    {
        g->fast_binding = std::make_unique<Ort::IoBinding>(*I.fast);
        const int64_t fseg = g->fast_seg;
        const std::array<int64_t, 4> fcache_shape{1, I.fast_n_local_heads, I.num_codebooks,
                                                  I.fast_head_dim};
        for (int64_t i = 0; i < I.num_fast_layers; ++i) {
            auto kv = CreateFP16Tensor(g->cuda_mem,
                                       static_cast<uint16_t*>(g->fast_kv[static_cast<size_t>(2 * i)]),
                                       static_cast<size_t>(fseg), fcache_shape.data(), 4);
            auto vv = CreateFP16Tensor(g->cuda_mem,
                                       static_cast<uint16_t*>(g->fast_kv[static_cast<size_t>(2 * i + 1)]),
                                       static_cast<size_t>(fseg), fcache_shape.data(), 4);
            g->fast_binding->BindInput(("cache_key_" + std::to_string(i)).c_str(), kv);
            g->fast_binding->BindInput(("cache_value_" + std::to_string(i)).c_str(), vv);
        }
        const Ort::MemoryInfo cpu_mem =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        g->fast_binding->BindOutput("logits", cpu_mem);
        // fast 单步:delta 输出形状固定 [1, fast_n_local_heads, 1, fast_head_dim](单 token),
        // 可在建 state 时 Value-绑一次到 scratch(ORT 1.29 无 4 参 BindOutput)。
        const std::array<int64_t, 4> fdshape{1, I.fast_n_local_heads, 1, I.fast_head_dim};
        const size_t fdcount = static_cast<size_t>(I.fast_n_local_heads * 1 * I.fast_head_dim);
        for (int64_t i = 0; i < I.num_fast_layers; ++i) {
            auto kout = CreateFP16Tensor(g->cuda_mem, g->fast_scratch[static_cast<size_t>(2 * i)],
                                         fdcount, fdshape.data(), 4);
            auto vout = CreateFP16Tensor(g->cuda_mem,
                                         g->fast_scratch[static_cast<size_t>(2 * i + 1)],
                                         fdcount, fdshape.data(), 4);
            g->fast_binding->BindOutput(("key_delta_" + std::to_string(i)).c_str(), kout);
            g->fast_binding->BindOutput(("value_delta_" + std::to_string(i)).c_str(), vout);
        }
    }
    return std::move(g);
}

bool Audio8Runtime::SlowStepGpu(Audio8GpuState* g, const std::vector<int64_t>& codes,
                                const std::vector<int64_t>& positions,
                                std::vector<float>* last_logits,
                                std::vector<uint16_t>* last_hidden) {
    if (!impl_ || !g || !impl_->slow || !last_logits || !last_hidden) return false;
    Impl& I = *impl_;
    const int64_t T = static_cast<int64_t>(positions.size());
    const int64_t rows = num_codebooks_ + 1;
    if (static_cast<int64_t>(codes.size()) != rows * T) return false;
    if (static_cast<int64_t>(I.slow_logits_size) <= 0) I.slow_logits_size = 4097;
    const int64_t seg = g->slow_seg;
    std::array<int64_t, 3> codes_shape{1, rows, T};
    std::array<int64_t, 1> pos_shape{T};
    auto cval = Ort::Value::CreateTensor<int64_t>(
        I.meminfo, const_cast<int64_t*>(codes.data()), codes.size(), codes_shape.data(), 3);
    auto pval = Ort::Value::CreateTensor<int64_t>(
        I.meminfo, const_cast<int64_t*>(positions.data()), positions.size(), pos_shape.data(), 1);
    g->slow_binding->BindInput("codes", cval);
    g->slow_binding->BindInput("input_pos", pval);
    // delta 输出:每帧按 [1,h,T,d] 形状 Value 绑到 scratch(ORT 1.29 无 4 参 BindOutput)
    {
        const std::array<int64_t, 4> dshape{1, I.n_local_heads, T, I.head_dim};
        const size_t dcount = static_cast<size_t>(I.n_local_heads * T * I.head_dim);
        for (int64_t i = 0; i < I.num_layers; ++i) {
            auto kout = CreateFP16Tensor(g->cuda_mem, g->slow_scratch[static_cast<size_t>(2 * i)],
                                         dcount, dshape.data(), 4);
            auto vout = CreateFP16Tensor(g->cuda_mem,
                                         g->slow_scratch[static_cast<size_t>(2 * i + 1)],
                                         dcount, dshape.data(), 4);
            g->slow_binding->BindOutput(("key_delta_" + std::to_string(i)).c_str(), kout);
            g->slow_binding->BindOutput(("value_delta_" + std::to_string(i)).c_str(), vout);
        }
    }
    try {
        I.slow->Run(Ort::RunOptions{nullptr}, *g->slow_binding);
    } catch (const Ort::Exception& e) {
        MD_LOG_ERROR << "audio8: slow gpu run failed: " << e.what() << std::endl;
        return false;
    }
    const auto outs = g->slow_binding->GetOutputValues();
    if (outs.size() < 2 + 2 * static_cast<size_t>(I.num_layers)) return false;
    {
        const float* logits = outs[0].GetTensorData<float>();
        last_logits->assign(logits, logits + I.slow_logits_size);
    }
    {
        const uint16_t* hidden = outs[1].GetTensorData<uint16_t>();
        last_hidden->assign(hidden, hidden + static_cast<size_t>(I.fast_dim));
    }
    // delta(已写进 scratch)按 strided 布局拷入 cache 槽位;失败 fail-closed(F3)
    const int64_t pos0 = positions[0];
    for (int64_t i = 0; i < I.num_layers; ++i) {
        for (int64_t h = 0; h < I.n_local_heads; ++h) {
            for (int64_t t = 0; t < T; ++t) {
                const int64_t dst = (h * I.max_seq_len + pos0 + t) * I.head_dim * 2;
                const int64_t src = ((h * T + t) * I.head_dim) * 2;
                const cudaError_t ek = cudaMemcpy(
                    static_cast<char*>(g->slow_kv[static_cast<size_t>(2 * i)]) + dst,
                    static_cast<char*>(g->slow_scratch[static_cast<size_t>(2 * i)]) + src,
                    static_cast<size_t>(I.head_dim) * 2, cudaMemcpyDeviceToDevice);
                if (ek != cudaSuccess) {
                    MD_LOG_ERROR << "audio8: slow gpu cache D2D key copy failed: "
                                 << cudaGetErrorString(ek) << std::endl;
                    return false;
                }
                const cudaError_t ev = cudaMemcpy(
                    static_cast<char*>(g->slow_kv[static_cast<size_t>(2 * i + 1)]) + dst,
                    static_cast<char*>(g->slow_scratch[static_cast<size_t>(2 * i + 1)]) + src,
                    static_cast<size_t>(I.head_dim) * 2, cudaMemcpyDeviceToDevice);
                if (ev != cudaSuccess) {
                    MD_LOG_ERROR << "audio8: slow gpu cache D2D value copy failed: "
                                 << cudaGetErrorString(ev) << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

bool Audio8Runtime::FastStepGpu(Audio8GpuState* g, int64_t token, bool use_hidden, int64_t position,
                                const std::vector<uint16_t>& slow_hidden,
                                std::vector<float>* last_logits) {
    if (!impl_ || !g || !impl_->fast || !last_logits) return false;
    if (g->fast_binding == nullptr) return false;
    Impl& I = *impl_;
    if (static_cast<int64_t>(slow_hidden.size()) != I.fast_dim) return false;
    const std::array<int64_t, 3> hidden_shape{1, 1, I.fast_dim};
    auto hval = CreateFP16Tensor(I.meminfo, const_cast<uint16_t*>(slow_hidden.data()),
                                 slow_hidden.size(), hidden_shape.data(), 3);
    const std::array<int64_t, 2> token_shape{1, 1};
    const std::array<int64_t, 1> scalar_shape{1};
    std::array<int64_t, 1> token_val{token};
    auto tval = Ort::Value::CreateTensor<int64_t>(I.meminfo, token_val.data(), token_val.size(),
                                                  token_shape.data(), 2);
    std::array<uint8_t, 1> use_val{static_cast<uint8_t>(use_hidden ? 1 : 0)};
    auto uval = CreateBoolTensor(I.meminfo, use_val.data(), 1, scalar_shape.data(), 1);
    std::array<int64_t, 1> pos_val{position};
    auto pval = Ort::Value::CreateTensor<int64_t>(I.meminfo, pos_val.data(), pos_val.size(),
                                                  scalar_shape.data(), 1);
    g->fast_binding->BindInput("slow_hidden", hval);
    g->fast_binding->BindInput("token_id", tval);
    g->fast_binding->BindInput("use_slow_hidden", uval);
    g->fast_binding->BindInput("input_pos", pval);
    try {
        I.fast->Run(Ort::RunOptions{nullptr}, *g->fast_binding);
    } catch (const Ort::Exception& e) {
        MD_LOG_ERROR << "audio8: fast gpu run failed: " << e.what() << std::endl;
        return false;
    }
    const auto outs = g->fast_binding->GetOutputValues();
    if (outs.size() < 1 + 2 * static_cast<size_t>(I.num_fast_layers)) return false;
    {
        const float* logits = outs[0].GetTensorData<float>();
        last_logits->assign(logits, logits + static_cast<size_t>(4096));
    }
    // delta 已写进 fast_scratch(形状 [1, h, 1, d]),按 strided 布局拷入 fast cache 槽位;
    // 失败 fail-closed(F3 与 slow 路径一致)。
    for (int64_t i = 0; i < I.num_fast_layers; ++i) {
        for (int64_t h = 0; h < I.fast_n_local_heads; ++h) {
            const int64_t dst = (h * I.num_codebooks + position) * I.fast_head_dim * 2;
            const int64_t src = h * I.fast_head_dim * 2;
            const cudaError_t ek = cudaMemcpy(
                static_cast<char*>(g->fast_kv[static_cast<size_t>(2 * i)]) + dst,
                static_cast<char*>(g->fast_scratch[static_cast<size_t>(2 * i)]) + src,
                static_cast<size_t>(I.fast_head_dim) * 2, cudaMemcpyDeviceToDevice);
            if (ek != cudaSuccess) {
                MD_LOG_ERROR << "audio8: fast gpu cache D2D key copy failed: "
                             << cudaGetErrorString(ek) << std::endl;
                return false;
            }
            const cudaError_t ev = cudaMemcpy(
                static_cast<char*>(g->fast_kv[static_cast<size_t>(2 * i + 1)]) + dst,
                static_cast<char*>(g->fast_scratch[static_cast<size_t>(2 * i + 1)]) + src,
                static_cast<size_t>(I.fast_head_dim) * 2, cudaMemcpyDeviceToDevice);
            if (ev != cudaSuccess) {
                MD_LOG_ERROR << "audio8: fast gpu cache D2D value copy failed: "
                             << cudaGetErrorString(ev) << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool Audio8Runtime::CopyGpuCacheToHost(Audio8GpuState* g, int kind, uint16_t* dst) {
    if (!g || !dst) return false;
    if (kind == 0) {
        uint16_t* out = dst;
        for (auto* p : g->slow_kv) {
            const cudaError_t ce = cudaMemcpy(out, p, static_cast<size_t>(g->slow_seg) * 2,
                                              cudaMemcpyDeviceToHost);
            if (ce != cudaSuccess) {
                MD_LOG_ERROR << "audio8: copy slow cache to host failed: "
                             << cudaGetErrorString(ce) << std::endl;
                return false;
            }
            out += static_cast<size_t>(g->slow_seg);
        }
    } else {
        uint16_t* out = dst;
        for (auto* p : g->fast_kv) {
            const cudaError_t ce = cudaMemcpy(out, p, static_cast<size_t>(g->fast_seg) * 2,
                                              cudaMemcpyDeviceToHost);
            if (ce != cudaSuccess) {
                MD_LOG_ERROR << "audio8: copy fast cache to host failed: "
                             << cudaGetErrorString(ce) << std::endl;
                return false;
            }
            out += static_cast<size_t>(g->fast_seg);
        }
    }
    return true;
}

#else  // !WITH_GPU
Audio8Runtime::Audio8GpuStatePtr Audio8Runtime::MakeGpuState(Device, int32_t) { return {}; }
bool Audio8Runtime::SlowStepGpu(Audio8GpuState*, const std::vector<int64_t>&,
                                const std::vector<int64_t>&, std::vector<float>*,
                                std::vector<uint16_t>*) { return false; }
bool Audio8Runtime::FastStepGpu(Audio8GpuState*, int64_t, bool, int64_t,
                                const std::vector<uint16_t>&, std::vector<float>*) { return false; }
bool Audio8Runtime::CopyGpuCacheToHost(Audio8GpuState*, int, uint16_t*) { return false; }
#endif  // WITH_GPU

}  // namespace modeldeploy::audio::tts::audio8
