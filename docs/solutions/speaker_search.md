# SpeakerSearch — 说话人检索 / 声纹库

> 架构层：`modeldeploy::audio::solution`
> 源码：`csrc/audio/solutions/speaker_search.h` / `.cpp`
> 示例：`examples/demo_speaker/demo_speaker.cpp`

## 职责

内存说话人声纹库：注册（enroll）带标签的 embedding 后，对查询 embedding 做 **top-k 余弦相似度检索**，返回最相似标签与得分。

## 应用场景

- 说话人确认与识别（声纹 1:N 检索）
- 门禁 / 客服身份匹配
- 注册后实时匹配

## 输入输出

- 输入：注册 `{label, embedding}` 与查询 `embedding`
- 输出：`{label, 相似度得分}` 列表（top-k）

## 关键 API

```cpp
using modeldeploy::audio::solution::SpeakerSearch;

SpeakerSearch s;
s.enroll(const std::string& label, const std::vector<float>& embedding);  // 注册/覆盖
std::vector<std::pair<std::string,float>> s.match(const std::vector<float>& embedding, int k = 1) const;
size_t s.size() const;   // 库中人声数量
```

## 原理与算法

转发到 `modeldeploy::audio::SpeakerGallery`（`label -> L2 归一化 embedding`）：

- `enroll`：`gallery_[label] = l2_normalize(embedding)`（同 label 覆盖）。
- `match`：查询向量先 L2 归一化，再与库中每个参考向量算**余弦相似度**，按得分降序排序取前 `k` 个（`k<=0` 时返回全部）。

> 注意：`match` 不设最低阈值，返回得分最高的 k 个；不持久化，仅内存。

## 典型用法

```cpp
// 用 SpeakerVerify 提取 embedding
SpeakerVerify model(model_file, option);
std::vector<float> emb;  model.predict(pcm, &emb);   // pcm 为 16k

SpeakerSearch s;
s.enroll("alice", embA);
s.enroll("bob",   embB);
auto top = s.match(emb_query, 1);                    // -> ("alice", score)
```

C API：`md_audio_speaker_search_enroll/match`；Python：`SpeakerSearch`（`enroll`/`match`）。
