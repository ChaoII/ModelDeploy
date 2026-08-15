using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.audio.tts
{
    public class Kokoro : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

        internal Kokoro(MDModel existing)
        {
            _model = existing;
        }

        public Kokoro(MDKokoroParameters parameters, RuntimeOption option)
        {
            _model = new MDModel();
            var nativeOption = option.ToNative();
            Utils.Check(md_create_kokoro_model(ref _model, ref parameters, ref nativeOption), "Create model");
        }

        public TtsResult Predict(string text, string voice, float speed)
        {
            MDTTSResult cResult = new MDTTSResult();

            using (var utf8Text = new Utf8String(text)) // 自动释放的封装
            {
                Utils.Check(md_kokoro_model_predict(ref _model, utf8Text.Ptr, voice, speed, ref cResult), "Predict");
            }

            TtsResult result = TtsResult.FromNative(cResult);
            md_free_kokoro_result(ref cResult);
            return result;
        }

        public static void WriteWav(TtsResult result, string outputPath)
        {
            MDTTSResult cResult = result.ToNative();
            try
            {
                Utils.Check(md_write_wav(ref cResult, outputPath), "WriteWav");
            }
            finally
            {
                md_free_kokoro_result(ref cResult);
            }
        }

        public Kokoro Clone()
        {
            var clone = new MDModel();
            Utils.Check(md_clone_model(ref clone, ref _model), "Clone Kokoro model");
            return new Kokoro(clone);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                md_free_kokoro_model(ref _model);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~Kokoro() => Dispose();






    }
}
