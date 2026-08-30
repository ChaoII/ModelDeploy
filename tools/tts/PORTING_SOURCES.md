# TTS 模型下载 / 移植对照源清单

任务 Task0：真实下载 Audio8 与 Qwen3-TTS-12Hz-0.6B 的 ONNX 模型 + 拉取移植对照源码。
模型根目录：`C:\ai\models\modeldeploy\tts\`（git 忽略，不入库）。

## 1. 下载脚本

- `tools/tts/download_tts_models.ps1`（本仓库内，唯一入库的下载产物）

网络经验：
- 本机 `huggingface-cli` 已废弃（只打印 "deprecated, use hf" 提示、不再执行任何下载），改用新版 `hf` CLI（v1.26.0）成功。
- 本地代理 `127.0.0.1:10809` 可用；`huggingface_hub`/`modelscope`/`git clone` 失败时以直连重试。
- ModelScope 可正常下载（Audio8 正是经 ModelScope 兜底成功）；`zukky/Qwen3-TTS-ONNX-DLL` 在 ModelScope 不存在（404）。

## 2. Audio8 对照源（Python, onnxruntime）

仓库：`Audio8-AI/Audio8_TTS`（默认分支 master，本机 clone HEAD=`55c1260`，`--depth 1`）
本地路径：`C:\ai\models\modeldeploy\tts\_src_audio8\`

关键四文件（绝对路径）：
- `C:\ai\models\modeldeploy\tts\_src_audio8\onnx_runtime\arktts_runtime\runtime.py`
- `C:\ai\models\modeldeploy\tts\_src_audio8\onnx_runtime\arktts_runtime\prompt.py`
- `C:\ai\models\modeldeploy\tts\_src_audio8\onnx_runtime\arktts_runtime\voices.py`
- `C:\ai\models\modeldeploy\tts\_src_audio8\onnx_runtime\arktts_runtime\registration.py`

## 3. Qwen3-TTS 移植对照源（sherpa-onnx 产出 NN 实现 + 导出脚本）

仓库：`HeiSir2014/sherpa-onnx`（分支 develop，本机 clone HEAD=`36d2e804` `Add Qwen3-TTS streaming support with async decode pipeline`）
本地路径：`C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\`

- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\sherpa-onnx\csrc\offline-tts-qwen3-impl.cc`
- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\sherpa-onnx\csrc\offline-tts-qwen3-model.h`
- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\sherpa-onnx\csrc\offline-tts-qwen3-model.cc`
- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\sherpa-onnx\csrc\offline-tts-qwen3-model-config.h`
- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\sherpa-onnx\csrc\offline-tts-qwen3-model-config.cc`
- `C:\ai\models\modeldeploy\tts\_src_sherpa_qwen3\scripts\qwen3-tts\export-onnx.py`

sherpa 侧 9 个子模型名（offline-tts-qwen3-model-config.h 确认）：
`text_project / codec_embed / code_predictor_embed / code_predictor / talker_prefill / talker_decode / speaker_encoder / tokenizer12hz_encode / tokenizer12hz_decode`
+ 可选 `tokenizer12hz_decode_stream`；tokenizer 用目录（vocab.json + merges.txt 即可，不需要 tokenizer.json）。

## 4. Audio8 模型文件清单（`audio8_preview\`）

仓库：`Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4`（HF；ModelScope 兜底下载）。共 16 文件。

| 文件 | 大小 (B) | MD5 |
|------|----------|-----|
| fast_ar_int4.onnx | 156,318 | 0349294D6F37DDE70364B05571A8DD4E |
| fast_ar_int4.onnx.data | 35,055,104 | B5FDE6D7FECC3061BB87AC454E02BE51 |
| slow_ar_int4.onnx | 900,218 | 5843BE93D81EDAD5AADA037B7982124A |
| slow_ar_int4.onnx.data | 290,267,090 | C34B2DFE219A83BE3F8F3D83B0887054 |
| codec_decoder_fp16.onnx | 594,319 | 8193834F546426762AF67DC9CDD030DF |
| codec_decoder_fp16.onnx.data | 260,741,440 | 266D34C511D3E0B47E36F69F00A6A863 |
| registration\codec_encoder_fp16.onnx | 940,787 | C945A093F93312DDAB993626C41C7E43 |
| registration\codec_encoder_fp16.onnx.data | 414,425,088 | 7B383035BE8BBAF147C8D73C6716561E |
| registration\registration_manifest.json | 165 | 262ACEB2253E8CB4F239F0CF262A29A4 |
| runtime_manifest.json | 1,080 | C5B7CC500AA26C433F02ABEFEA990644 |
| tokenizer\tokenizer.json | 12,217,872 | 3BCA944C013929D598EE757B267D8DA4 |
| config.json | 3 | 8A80554C91D9FCA8ACB82F023DE02F11 |
| configuration.json | 72 | B536DE665C0C52677F7C401D45D9D62B |
| README.md | 8,354 | 3165E6A7FFE48936BBD0B517A00F0C71 |

runtime_manifest.json 关键字段（前 10）：
`model_family=audio8_tts`、`activation_dtype=float16`、`slow_logits_layout=semantic_then_eos`、
`slow_logits_size=4097`、`kv_attention_layout=valid_prefix`、`max_seq_len=2048`、`num_layers=24`、
`num_fast_layers=4`、`num_codebooks=10`、`n_local_heads=2`，
另有 `semantic_begin_id=151678`、`semantic_end_id=155773`、`eos_token_id=151645`、
`codec_sample_rate=44100`、`default_precision=int4`、`available_precisions=[int4]`、
`default_codec_precision=fp16`、`available_codec_precisions=[fp16]`。

## 5. Qwen3 模型文件清单（`qwen3_tts_0.6b\`）

仓库：`zukky/Qwen3-TTS-ONNX-DLL`（HF 公开社区仓库），ONNX 位于 `onnx_kv_06b/`（0.6B / 12Hz），
tokenizer 位于 `models/Qwen3-TTS-12Hz-0.6B-Base/`。**已拿到完整可用 ONNX 全套，无需二次转换。**

| 文件 | 大小 (B) | MD5 |
|------|----------|-----|
| onnx_kv_06b\text_project.onnx | 1,269,839,113 | 0D1137D0E2E028A621C04F4F887921D3 |
| onnx_kv_06b\codec_embed.onnx | 12,583,154 | F61792AAD8759EA1003AF002BA4D0D1B |
| onnx_kv_06b\code_predictor_embed.onnx | 125,829,841 | 9D2A3F29B909CB169789C041704D4295 |
| onnx_kv_06b\code_predictor.onnx | 440,684,435 | B14ED3DCD96B8F60BA968D0467D7064A |
| onnx_kv_06b\talker_prefill.onnx | 1,776,976,739 | 39C00BEE6A354F45E7F8373D1A42E54B |
| onnx_kv_06b\talker_decode.onnx | 1,776,980,336 | DC4105D587C89BB3951829C865A9F127 |
| onnx_kv_06b\speaker_encoder.onnx | 35,625,029 | 6E7CDCB9A2D5D282FACE4C58FFC1DBEB |
| onnx_kv_06b\tokenizer12hz_encode.onnx | 192,844,705 | 1DA8760E2E981DFFCAF119B3CCF3BBA5 |
| onnx_kv_06b\tokenizer12hz_decode.onnx | 456,825,792 | 0891498F65FDD43CEF262589872C5FA4 |
| onnx_kv_06b\tokenizer12hz_decode_1024.onnx | 456,825,792 | 0891498F65FDD43CEF262589872C5FA4 |
| models\Qwen3-TTS-12Hz-0.6B-Base\config.json | 4,494 | F48B7DA70745E50D5F27CB0CB8E7AF9E |
| models\Qwen3-TTS-12Hz-0.6B-Base\merges.txt | 1,671,839 | E78882C2E224A75FA8180EC610BAE243 |
| models\Qwen3-TTS-12Hz-0.6B-Base\tokenizer_config.json | 7,344 | 873B59BF6DF5B60F35E2C14EDC1B33AE |
| models\Qwen3-TTS-12Hz-0.6B-Base\vocab.json | 2,776,833 | 613B8E4A622C4A2C90E9E1245FC540D6 |

注：`tokenizer12hz_decode.onnx` 与 `tokenizer12hz_decode_1024.onnx` 大小、MD5 完全相同（仓库内即重复份）。

## 6. 验证结论

- 全部 10 个 Qwen3 ONNX 经 onnxruntime 1.20.1 CPU 直接 `InferenceSession` 加载成功。
- 全部 4 个 Audio8 ONNX（含外部权重 .onnx.data）加载成功。
- Qwen3 **已拿到社区 ONNX 全套**（talker_prefill.onnx 等，sherpa 命名），无需 safetensors 兜底 / 无需 export-onnx.py 二次转换。
