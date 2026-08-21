using System;
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
}

namespace ModelDeploy
{
    /// <summary>把托管 string 以 UTF-8 字节分配为本机缓冲区（netstandard2.0 无 LPUTF8Str，手动封送）。</summary>
    internal static class Utf8
    {
        public static IntPtr Alloc(string s)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(s);
            var ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(bytes.Length + 1);
            System.Runtime.InteropServices.Marshal.Copy(bytes, 0, ptr, bytes.Length);
            System.Runtime.InteropServices.Marshal.WriteByte(ptr, bytes.Length, 0);
            return ptr;
        }

        public static void Free(IntPtr ptr) => System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
}
