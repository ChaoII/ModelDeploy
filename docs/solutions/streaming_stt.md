# StreamingSTT — 流式语音识别

> 架构层：`modeldeploy::audio::solution`
> 源码：`csrc/audio/solutions/streaming_stt.h` / `.cpp`
> 示例：`examples/demo_audio_solutions/demo_stream_stt.cpp`

## 职责

把分块推入的音频流经 **VAD 切出语音段**，逐段调用 ASR 转写回调，通过回调**实时输出识别文本**。

## 应用场景

- 实时语音转写（会议、语音客服、直播字幕）
- 语音自动断句与字幕生成

## 输入输出

- 输入：分块音频 `push(data, sr)`（流式推入）
- 输出：经 `on_text` 回调实时输出的识别文本

## 关键 API

```cpp
using modeldeploy::audio::solution::StreamingSTT;
using TranscribeFn = std::function<std::string(const std::vector<float>&, int sr)>;

StreamingSTT stt(TranscribeFn on_text = nullptr,     // 识别结果回调
                 std::function<void(const std::string&)> transcribe = nullptr, // 转写回调
                 int sample_rate = 16000);           // VAD 工作采样率
stt.push(const std::vector<float>& data, int sr);    // 推入一段音频
stt.set_on_text(cb);  stt.set_transcribe(TranscribeFn cb);
stt.run_once();                                      // 消费当前就绪语音段并触发回调
stt.finish();                                        // 流结束，转写末尾语音段
static TranscribeFn sense_voice(asr::SenseVoice& model);  // 包装真实 SenseVoice
```

## 原理与算法

1. `push()`：采样率与构造不一致时**置空分段结果而不重采样**，但一律 `vad_.feed(data)`。
2. `run_once()`：`vad_.consume()` 取出"当前已就绪"的语音段（消费后移除前缀、仅保留末尾尾随静音），对每段 `transcribe_(seg.samples, sr_)` 取文本，再触发 `on_text_(text)`。可通过多次 `push + run_once` 流式推进。
3. `finish()`：`vad_.finish()` 追加 `sr*silence_ms/1000` 个静音使末尾开口语音段闭合切出，再 `run_once()` 转写最后一截。
4. `sense_voice()`：闭包调用 `model.predict(data, &text)`，成功返回文本，否则空串。

## 典型用法

```cpp
StreamingSTT stt(
    [](const std::string& text){ printf("[STT] %s\n", text.c_str()); });
stt.set_transcribe(StreamingSTT::sense_voice(*sv));   // sv 为 SenseVoice 实例

const size_t chunk = sr / 4;                          // 250ms 一块
for (size_t off = 0; off < audio.size(); off += chunk) {
    stt.push({audio.begin()+off, audio.begin()+off+chunk}, sr);
    stt.run_once();
}
stt.finish();                                         // 流结束
```

> VAD 与 ASR 均按固定采样率（默认 16k）工作；仅分段（不配 transcribe）时回调收到空串。
