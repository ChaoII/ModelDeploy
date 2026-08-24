# SpeakerDiarization — 说话人日志

> 架构层：`modeldeploy::audio::solution`
> 源码：`csrc/audio/solutions/speaker_diarization.h` / `.cpp`
> 示例：`examples/demo_audio_solutions/demo_diarization.cpp`

## 职责

对一段音频先做 **VAD 分段**，再对每段提取说话人 embedding 并**聚类**，输出"时间区间 → 说话人 ID"的分段结果（说话人日志）。

## 应用场景

- 会议录音"谁在什么时候说话"分析
- 客服通话角色分离与话术分析
- 仅需语音活动检测（VAD 分段）也可独立使用

## 输入输出

- 输入：16k float PCM 音频 `std::vector<float>`
- 输出：`std::vector<Segment>`（`{start_ms, end_ms, speaker_id}`）

## 关键 API

```cpp
using modeldeploy::audio::solution::SpeakerDiarization;
struct Segment { int start_ms; int end_ms; int speaker_id; };

SpeakerDiarization dia;
bool ok = dia.run(const std::vector<float>& audio, std::vector<Segment>* out,
                  const EmbedFn& embed = nullptr, float threshold = 0.7f);

using EmbedFn = std::function<std::vector<float>(const std::vector<float>&, int)>;
std::vector<int> assign_speakers(const std::vector<std::vector<float>>& embeddings,
                                 float threshold = 0.7f) const;
static EmbedFn ecapa_embedder(speaker_verify::SpeakerVerify& model);  // 包装真实 ECAPA
```

## 原理与算法

1. `run()` 以 **16000 Hz** 构造 `tool::VadSegment`，`feed(audio)` 后取得语音段列表。
2. 若传入 `embed`：对每段取 embedding；空 embedding（模型未初始化或推理失败）该段标 `-1`。只要存在任意有效 embedding 就调用 `assign_speakers()` 聚类。
3. `assign_speakers()` 为**在线 / 增量聚类（顺序贪心）**：
   - 对待分 embedding 先 L2 归一化；
   - 与已有各簇形心算**余弦相似度**取最大者；
   - 若最大相似度 `>= threshold`（默认 0.7）归入该簇，并以衰减加权更新形心 `c = c*(1-w) + q*w`（`w=0.5`）；
   - 否则新建簇，ID 递增。
4. 把 `VadSegment::Seg` 的 `start_ms/end_ms` 连同聚类 `speaker_id` 填进 `Segment` 输出。
5. `ecapa_embedder()`：闭包调用 `model.predict(data, &emb)`，成功返回 embedding，否则空向量（触发 -1 路径）。

## 典型用法

```cpp
modeldeploy::RuntimeOption opt; opt.use_cpu();
auto sv = std::make_unique<...SpeakerVerify>(argv[2], opt);
auto embed = SpeakerDiarization::ecapa_embedder(*sv);

SpeakerDiarization dia;
std::vector<SpeakerDiarization::Segment> segs;
dia.run(audio, &segs, embed);                    // audio 为 16k float PCM
for (auto& s : segs)
    printf("[%d ms, %d ms] speaker=%d\n", s.start_ms, s.end_ms, s.speaker_id);
```

不传 embed（无模型）时仅做 VAD 分段说话人标 `-1`。聚类阈值 `threshold` 控制说话人合并松紧（示例常取 0.5）。
