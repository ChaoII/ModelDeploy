using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.asr;

public class FunASROffLine
{
    private MDModel model;

    public FunASROffLine(
        string model_dir,
        string vad_dir,
        string punct_dir,
        string itn_dir = "",
        string lm_dir = "",
        string hot_word_path = "",
        bool blade_disc = true,
        float global_beam = 3.0f,
        float lattice_beam = 3.0f,
        float am_scale = 10.0f,
        int fst_inc_wts = 20,
        int thread_num = 1,
        int batch_size = 4,
        bool use_gpu = false)
    {
        model = new MDModel();
        md_create_asr_offline_model(ref model,
            model_dir,
            vad_dir,
            punct_dir,
            itn_dir,
            lm_dir,
            hot_word_path,
            blade_disc,
            global_beam,
            lattice_beam,
            am_scale,
            fst_inc_wts,
            thread_num,
            batch_size,
            use_gpu);
    }

    ~FunASROffLine()
    {
        md_free_asr_offline_model(ref model);
    }

    public ASRResult Predict(string wavPath, int audioFs = 16000)
    {
        MDASRResult asr_result = new MDASRResult();
        md_asr_offline_model_predict(ref model, wavPath, ref asr_result, audioFs);
        ASRResult asrResult = ASRResult.FromMDASRResult(asr_result);
        md_free_asr_result(ref asr_result);
        return asrResult;
    }


    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_create_asr_offline_model(ref MDModel model,
        string model_dir,
        string vad_dir,
        string punct_dir,
        string itn_dir = "",
        string lm_dir = "",
        string hot_word_path = "",
        bool blade_disc = true,
        float global_beam = 3.0f,
        float lattice_beam = 3.0f,
        float am_scale = 10.0f,
        int fst_inc_wts = 20,
        int thread_num = 1,
        int batch_size = 4,
        bool use_gpu = false);


    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_asr_offline_model_predict(ref MDModel model, string wav_path,
        ref MDASRResult asr_result, int audio_fs = 16000);
    
    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void md_free_asr_result(ref MDASRResult asr_result);
    
    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void md_free_asr_offline_model(ref MDModel model);
}