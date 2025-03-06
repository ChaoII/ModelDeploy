using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.audio.tts
{
    public class Kokoro
    {
        private MDModel model;

        public Kokoro(MDKokoroParameters parameters)
        {
            model = new MDModel();
            md_create_kokoro_model(ref model, ref parameters);
        }

        ~Kokoro()
        {
            md_free_kokoro_model(ref model);
        }

        public void Predict(string text, int sid, float speed, string wavPath)
        {
            md_kokoro_model_predict(ref model, text, sid, speed, wavPath);
        }
        
        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_create_kokoro_model(ref MDModel model, ref MDKokoroParameters parameters);


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int md_kokoro_model_predict(ref MDModel model, string text, int sid, float speed,
            string wavPath);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_free_kokoro_model(ref MDModel model);
    }
}