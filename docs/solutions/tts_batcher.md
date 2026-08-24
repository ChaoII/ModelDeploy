# TTSBatcher — 文本转语音批处理与长文本分块

> 架构层：`modeldeploy::audio::solution`
> 源码：`csrc/audio/solutions/tts_batcher.h` / `.cpp`
> 示例：`examples/demo_audio_solutions/demo_tts_batch.cpp`

## 职责

维护一个待合成文本队列，批量出队交给合成回调；并提供**按字符上限切块的长文本合成 / 拼接**工具，避免超出 TTS 模型 token 上限被静默截断。

## 应用场景

- 批量 TTS 合成（批量文章 / 通知朗读、离线语音生成）
- 长文本连续朗读（自动切块拼接）
- 通过 Kokoro 支持中英文多语言合成

## 输入输出

- 输入：文本 `enqueue`
- 输出：`dequeue_all()` 返回 `std::vector<std::vector<float>>`（每条文本对应一段 float PCM 音频）

## 关键 API

```cpp
using SynthFn = std::function<std::vector<float>(const std::string&)>;  // 文本→音频

TTSBatcher b(SynthFn synth = nullptr);
b.enqueue(const std::vector<std::string>& texts);  // 批量入队
b.enqueue(const std::string& text);                // 单个入队
std::vector<std::vector<float>> b.dequeue_all();   // 出队并合成，清空队列
size_t b.pending() const;                          // 当前排队条数
b.set_synth(SynthFn synth);

static SynthFn kokoro_synth(tts::Kokoro& model, const std::string& voice, float speed = 1.0f);
static std::vector<float> synthesize_text(tts::Kokoro& model, const std::string& voice,
                                          float speed, const std::string& text, int max_chars = 120);
static std::vector<std::string> split_for_synthesis(const std::string& text, int max_chars);
```

## 原理与算法

1. 队列：`enqueue` 插入，`dequeue_all` 逐个 `synth_(t)`（无 synth 时每条输出空向量）后清空。
2. `split_for_synthesis()`（核心分块）：
   - 按 **UTF-8 字符**计数（首字节高位判断字符字节数 1–4）；
   - 若总字符数 `<= max_chars` 直接整体返回；
   - 否则从窗口内查出**最后一个可断点**（`空格 \t . , ; ! ? 。 ， ； ！ ？ 、 \n \r`）处截断；找不到断点则硬切，**保证不丢字**。
3. `synthesize_text()`：模型未初始化返回空；否则分块后逐块 `model.predict(p, voice, speed, &audio)` 并拼接（某块失败跳过）。
4. `kokoro_synth()`：调 `synthesize_text(text)`（默认 `max_chars=120`）。

## 典型用法

```cpp
TTSBatcher batcher;
auto* kokoro = new Kokoro(model, tokens, lexicons, voices_bin, dict, dir, opt);
batcher.set_synth(TTSBatcher::kokoro_synth(*kokoro, "zf_001", 1.0f));

batcher.enqueue(text);                    // 长文本，内部按标点切块
auto batches = batcher.dequeue_all();     // 每条文本 -> 一段 float PCM
```

C API：`MD_AUDIO_TTS_BATCHER`；Python：`TTSBatcher`（`enqueue`/`dequeue_all`）。
