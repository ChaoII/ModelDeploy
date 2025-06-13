using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.audio.tts
{
    public class Kokoro : IDisposable
    {
        private MDModel _model;
        private bool _disposed;

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
            Utils.Check(md_write_wav(ref cResult, outputPath), "WriteWav");
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

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_kokoro_model(ref MDModel model, ref MDKokoroParameters parameters,
            ref MDRuntimeOption option);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_kokoro_model_predict(ref MDModel model, IntPtr text, string voice, float speed,
            ref MDTTSResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_write_wav(ref MDTTSResult result, string outputPath);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_free_kokoro_result(ref MDTTSResult result);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_kokoro_model(ref MDModel model);
    }
}