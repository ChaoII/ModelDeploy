using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

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
        
        



    }
}
