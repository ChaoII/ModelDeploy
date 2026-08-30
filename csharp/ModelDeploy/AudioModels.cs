using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.Results;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.Models
{
    /// <summary>SenseVoice ASR。modelPath 格式: model.onnx|tokens.txt</summary>
    public sealed class SenseVoiceModel : BaseModel
    {
        private SenseVoiceModel(IntPtr handle) : base(MDModelKind.MD_MODEL_ASR, handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public SenseVoiceModel Clone() => new SenseVoiceModel(CloneNative());

        public SenseVoiceModel(string modelPath, RuntimeOption opt = null)
            : base(MDModelKind.MD_MODEL_ASR, modelPath, opt) { }

        public AsrResult PredictWav(string wavPath)
        {
            var status = md_audio_asr_wav(_handle, wavPath, out var text);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"ASR predict failed: {GetLastError()}");
            return new AsrResult { Text = ResultReader.ReadString(text) ?? string.Empty };
        }

        public AsrResult Predict(float[] samples, int sampleRate)
        {
            var status = md_audio_asr(_handle, samples, new UIntPtr((uint)samples.Length), sampleRate, out var text);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"ASR predict failed: {GetLastError()}");
            return new AsrResult { Text = ResultReader.ReadString(text) ?? string.Empty };
        }

        static AsrResult ToResult(in MDAsrResult r) => new AsrResult
        {
            Text = ResultReader.ReadString(r.text) ?? string.Empty,
            Language = ResultReader.ReadString(r.language) ?? string.Empty,
            Emotion = ResultReader.ReadString(r.emotion) ?? string.Empty,
            Event = ResultReader.ReadString(r.@event) ?? string.Empty,
            Task = ResultReader.ReadString(r.task) ?? string.Empty,
            Itn = r.itn != 0,
            NoSpeech = r.nospeech != 0,
        };

        /// <summary>结构化识别（SenseVoice 复任务标签），从 wav 文件。</summary>
        public AsrResult PredictWavStructured(string wavPath)
        {
            var status = md_audio_asr_wav_result(_handle, wavPath, out var r);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"ASR structured failed: {GetLastError()}");
            return ToResult(in r);
        }

        /// <summary>结构化识别（SenseVoice 复任务标签），从 PCM 浮点采样。</summary>
        public AsrResult PredictStructured(float[] samples, int sampleRate)
        {
            var status = md_audio_asr_result(_handle, samples, new UIntPtr((uint)samples.Length), sampleRate, out var r);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"ASR structured failed: {GetLastError()}");
            return ToResult(in r);
        }
    }

    /// <summary>Kokoro TTS。modelPath 格式: model.onnx|tokens.txt|lex_en.txt|lex_zh.txt|voices.bin|jieba_dir|norm_dir</summary>
    public sealed class KokoroModel : BaseModel
    {
        private KokoroModel(IntPtr handle) : base(MDModelKind.MD_MODEL_TTS, handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public KokoroModel Clone() => new KokoroModel(CloneNative());

        public KokoroModel(string modelPath, RuntimeOption opt = null)
            : base(MDModelKind.MD_MODEL_TTS, modelPath, opt) { }

        public TtsResult Predict(string text, string voice, float speed = 1.0f)
        {
            var textPtr = Utf8.Alloc(text);
            var voicePtr = Utf8.Alloc(voice);
            try
            {
                var status = md_audio_tts(_handle, textPtr, voicePtr, speed, out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS predict failed: {GetLastError()}");
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                Utf8.Free(textPtr);
                Utf8.Free(voicePtr);
            }
        }

        /// <summary>把音频写入 wav 文件。</summary>
        public void SaveWav(TtsResult result, string path)
        {
            var status = md_wav_save(result.Audio, new UIntPtr((uint)result.Audio.Length),
                result.SampleRate, path);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Wav save failed: {GetLastError()}");
        }
    }

    /// <summary>Audio8 TTS。modelDir 为模型目录（onnx 权重 + 配置）。</summary>
    public sealed class Audio8Model : BaseModel
    {
        private Audio8Model(IntPtr handle) : base(MDModelKind.MD_MODEL_TTS_AUDIO8, handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public Audio8Model Clone() => new Audio8Model(CloneNative());

        /// <summary>最近一次合成采样率（Predict/PredictStream 后刷新）。</summary>
        public int SampleRate { get; private set; }

        public Audio8Model(string modelDir, RuntimeOption opt = null)
            : base(MDModelKind.MD_MODEL_TTS_AUDIO8, modelDir, opt) { }

        public TtsResult Predict(string text, string voice, float speed = 1.0f)
        {
            var textPtr = Utf8.Alloc(text);
            var voicePtr = Utf8.Alloc(voice);
            try
            {
                var status = md_audio_tts(_handle, textPtr, voicePtr, speed, out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS predict failed: {GetLastError()}");
                SampleRate = sr;
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                Utf8.Free(textPtr);
                Utf8.Free(voicePtr);
            }
        }

        /// <summary>TTS 流式合成：逐块回调 onChunk(samples, progress)，同时返回整段音频。
        /// chunkFrames &lt;= 0 时等价一次性合成（单次回调整段）。
        /// Qwen3 mode B 的 AR 阶段会回调 progress 空块（n==0），本层跳过音频处理、仅上报进度。</summary>
        public TtsResult PredictStream(string text, string voice, float speed, int chunkFrames,
            Action<float[], float> onChunk)
        {
            if (onChunk == null) throw new ArgumentNullException(nameof(onChunk));
            var textPtr = Utf8.Alloc(text);
            var voicePtr = Utf8.Alloc(voice);
            MDTtsAudioCb cb = (samplesPtr, n, progress, userdata) =>
            {
                var target = GCHandle.FromIntPtr(userdata).Target as Action<float[], float>;
                if (target != null)
                {
                    // n==0 为 progress 空块（Qwen3 mode B AR 阶段）：跳过音频复制、仅上报进度。
                    if (n > 0 && samplesPtr != IntPtr.Zero)
                    {
                        var samples = new float[n];
                        Marshal.Copy(samplesPtr, samples, 0, n);
                        target(samples, progress);
                    }
                    else
                    {
                        target(Array.Empty<float>(), progress);
                    }
                }
                return 1;
            };
            var userdata = GCHandle.Alloc(onChunk);
            try
            {
                var status = md_audio_tts_stream(_handle, textPtr, voicePtr, speed, chunkFrames,
                    cb, GCHandle.ToIntPtr(userdata), out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS stream failed: {GetLastError()}");
                SampleRate = sr;
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                userdata.Free();
                Utf8.Free(textPtr);
                Utf8.Free(voicePtr);
                GC.KeepAlive(cb);
            }
        }
    }

    /// <summary>Qwen3 TTS（含声音克隆）。modelDir 为模型目录。</summary>
    public sealed class Qwen3TtsModel : BaseModel
    {
        private Qwen3TtsModel(IntPtr handle) : base(MDModelKind.MD_MODEL_TTS_QWEN3, handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public Qwen3TtsModel Clone() => new Qwen3TtsModel(CloneNative());

        public Qwen3TtsModel(string modelDir, RuntimeOption opt = null)
            : base(MDModelKind.MD_MODEL_TTS_QWEN3, modelDir, opt) { }

        public TtsResult Predict(string text, string voice, float speed = 1.0f)
        {
            var textPtr = Utf8.Alloc(text);
            var voicePtr = Utf8.Alloc(voice);
            try
            {
                var status = md_audio_tts(_handle, textPtr, voicePtr, speed, out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS predict failed: {GetLastError()}");
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                Utf8.Free(textPtr);
                Utf8.Free(voicePtr);
            }
        }

        /// <summary>TTS 流式合成：逐块回调 onChunk(samples, progress)，同时返回整段音频。
        /// chunkFrames &lt;= 0 时等价一次性合成（单次回调整段）。
        /// Qwen3 mode B 的 AR 阶段会回调 progress 空块（n==0），本层跳过音频处理、仅上报进度。</summary>
        public TtsResult PredictStream(string text, string voice, float speed, int chunkFrames,
            Action<float[], float> onChunk)
        {
            if (onChunk == null) throw new ArgumentNullException(nameof(onChunk));
            var textPtr = Utf8.Alloc(text);
            var voicePtr = Utf8.Alloc(voice);
            MDTtsAudioCb cb = (samplesPtr, n, progress, userdata) =>
            {
                var target = GCHandle.FromIntPtr(userdata).Target as Action<float[], float>;
                if (target != null)
                {
                    // n==0 为 progress 空块（Qwen3 mode B AR 阶段）：跳过音频复制、仅上报进度。
                    if (n > 0 && samplesPtr != IntPtr.Zero)
                    {
                        var samples = new float[n];
                        Marshal.Copy(samplesPtr, samples, 0, n);
                        target(samples, progress);
                    }
                    else
                    {
                        target(Array.Empty<float>(), progress);
                    }
                }
                return 1;
            };
            var userdata = GCHandle.Alloc(onChunk);
            try
            {
                var status = md_audio_tts_stream(_handle, textPtr, voicePtr, speed, chunkFrames,
                    cb, GCHandle.ToIntPtr(userdata), out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS stream failed: {GetLastError()}");
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                userdata.Free();
                Utf8.Free(textPtr);
                Utf8.Free(voicePtr);
                GC.KeepAlive(cb);
            }
        }

        /// <summary>声音克隆：refAudio 为参考音频 wav 路径，refText 为参考文本，lang 语言白名单
        /// （null/空串视为 "auto"），成功返回 24kHz 克隆音频。</summary>
        public TtsResult Clone(string text, string refAudio, string refText, string lang = null)
        {
            var textPtr = Utf8.Alloc(text);
            var refAudioPtr = Utf8.Alloc(refAudio);
            var refTextPtr = Utf8.Alloc(refText);
            var langPtr = Utf8.Alloc(lang);
            try
            {
                var status = md_audio_tts_qwen3_clone(_handle, textPtr, refAudioPtr, refTextPtr, langPtr,
                    out var sr, out var audio, out var n);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"TTS clone failed: {GetLastError()}");
                return new TtsResult
                {
                    Audio = ResultReader.ReadFloats(audio, n),
                    SampleRate = sr
                };
            }
            finally
            {
                Utf8.Free(textPtr);
                Utf8.Free(refAudioPtr);
                Utf8.Free(refTextPtr);
                Utf8.Free(langPtr);
            }
        }
    }
}

namespace ModelDeploy
{
    /// <summary>把托管 string 以 UTF-8 字节分配为本机缓冲区（netstandard2.0 无 LPUTF8Str，手动封送）。</summary>
    internal static class Utf8
    {
        public static IntPtr Alloc(string s)
        {
            if (string.IsNullOrEmpty(s)) return IntPtr.Zero;
            var bytes = System.Text.Encoding.UTF8.GetBytes(s);
            var ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(bytes.Length + 1);
            System.Runtime.InteropServices.Marshal.Copy(bytes, 0, ptr, bytes.Length);
            System.Runtime.InteropServices.Marshal.WriteByte(ptr, bytes.Length, 0);
            return ptr;
        }

        public static void Free(IntPtr ptr) => System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
}
