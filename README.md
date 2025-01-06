# WRZS
> [ !NOTE ]
> 
>
bash
```bash
# funasr-onnx-2pass
demo.exe ^
--vad-dir D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx ^
--online-model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx ^
--model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx ^
--lm-dir D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst  ^
--punc-dir D:/funasr-runtime-resources/models/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx ^
--itn-dir D:/funasr-runtime-resources/models/fst_itn_zh ^
--mode online ^
--wav-path D:/funasr-runtime-resources/vad_example.wav
```



```bash
#funasr-onnx-offline
demo.exe ^
--model-dir D:/funasr-runtime-resources/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx ^
--punc-dir D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx ^
--vad-dir D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx ^
--vad-quant true ^
--punc-quant true ^
--wav-path D:/funasr-runtime-resources/vad_example.wav
```