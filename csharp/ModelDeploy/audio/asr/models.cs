using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.audio.asr
{
    public class SenseVoice : IDisposable
    {
        private MDModel model;
        private bool _disposed;

        internal SenseVoice(MDModel existing)
        {
            model = existing;
        }

        public SenseVoice(MDSenseVoiceParameters parameters)
        {
            model = new MDModel();
            md_create_sense_voice_model(ref model, ref parameters);
        }

        ~SenseVoice()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public SenseVoice Clone()
        {
            var clone = new MDModel();
            var ret = md_clone_model(ref clone, ref model);
            if (ret != 0)
                throw new InvalidOperationException($"Clone SenseVoice model failed (error code: {ret})");
            return new SenseVoice(clone);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed) return;
            _disposed = true;
            md_free_sense_voice_model(ref model);
        }

        public ASRResult Predict(string wavPath, int audioFs = 16000)
        {
            MDASRResult asr_result = new MDASRResult();
            md_sense_voice_model_predict(ref model, wavPath, ref asr_result, audioFs);
            ASRResult asrResult = ASRResult.FromMDASRResult(asr_result);
            md_free_sense_voice_result(ref asr_result);
            return asrResult;
        }
        
        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_sense_voice_model(ref MDModel model, ref MDSenseVoiceParameters parameters);
        
        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_sense_voice_model_predict(ref MDModel model, string wav_path,
            ref MDASRResult asr_result, int audio_fs = 16000);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_sense_voice_result(ref MDASRResult asr_result);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_sense_voice_model(ref MDModel model);

        [DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_clone_model(ref MDModel model, ref MDModel from);
    }
}