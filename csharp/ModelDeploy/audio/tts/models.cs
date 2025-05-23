using System;
using System.Runtime.InteropServices;
using System.Text;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ModelDeploy.audio.tts
{
    public class Kokoro
    {
        private MDModel _model;

        public Kokoro(MDKokoroParameters parameters)
        {
            _model = new MDModel();
            md_create_kokoro_model(ref _model, ref parameters);
        }

        ~Kokoro()
        {
            md_free_kokoro_model(ref _model);
        }

        public TTSResult Predict(string text, string voice, float speed)
        {
            // 注意一定要转化为原始的utf8字符串（char）指针
            MDTTSResult cResult = new MDTTSResult();
            var ptr = Utils.ConvertStringToHGlobalUtf8(text);
            int ret = md_kokoro_model_predict(ref _model, ptr, voice, speed, ref cResult);
            // 注意使用完后需要手动释放开辟的地址
            Marshal.FreeHGlobal(ptr);
            if (ret != 0)
            {
                throw new Exception("kokoro model predict failed error code is: " + ret);
            }

            TTSResult ocrResult = TTSResult.FromMDTTSResult(cResult);
            md_free_kokoro_result(ref cResult);
            return ocrResult;
        }

        public static void WriteWav(TTSResult result, string outputPath)
        {
            MDTTSResult cResult = TTSResult.ToMDTTSResult(result);
            int ret = md_write_wav(ref cResult, outputPath);
            if (ret != 0)
            {
                throw new Exception("md_write_wav failed error code is: " + ret);
            }
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_kokoro_model(ref MDModel model, ref MDKokoroParameters parameters);


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