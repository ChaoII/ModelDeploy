# WRZS
> [ !NOTE ]
> 
>
bash
```bash
# funasr-onnx-2pass
demo.exe ^
--model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx ^
--online-model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx ^
--vad-dir D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx ^
--punc-dir D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx ^
--lm-dir D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst  ^
--itn-dir D:/funasr-runtime-resources/models/fst_itn_zh ^
--mode online ^
--wav-path D:/funasr-runtime-resources/vad_example.wav
```



```bash
#funasr-onnx-offline
demo-asr.exe ^
--model-dir D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx ^
--vad-dir D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx ^
--punc-dir D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx ^
--vad-quant true ^
--punc-quant true ^
--wav-path D:/funasr-runtime-resources/vad_example.wav
```

```bash
asr_wss_server.exe ^
--model-dir  D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx ^
--online-model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx ^
--vad-dir D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx ^
--punc-dir D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx ^
--lm-dir D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst ^
--itn-dir D:/funasr-runtime-resources/models/fst_itn_zh ^
--cert-file ssl_key/server.crt ^
--key-file ssl_key/server.key
```