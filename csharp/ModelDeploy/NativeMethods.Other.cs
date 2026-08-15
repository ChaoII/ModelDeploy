//
// OCR / LPR / 行人属性 pipeline / 音频 P/Invoke。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    internal static partial class NativeMethods
    {
        #region OCR PaddleOCR

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_ocr_model(ref MDModel model, ref MDOCRModelParameters parameters,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_det_set_max_side_len(ref MDModel model, int maxSideLen = 960);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_det_set_db_thresh(ref MDModel model, double dbThresh = 0.3);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_det_set_db_box_thresh(ref MDModel model, double dbBoxThresh = 0.6);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_det_db_unclip_ratio(ref MDModel model, double dbUnclipRatio = 1.5);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_det_db_set_use_dilation(ref MDModel model, int useDilation = 0);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_model_predict(ref MDModel model, ref MDImage image, ref MDOCRResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDRect md_get_text_position(ref MDModel model, ref MDImage image, string text);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_ocr_result(ref MDImage image, ref MDOCRResults results, string fontPath,
            int fontSize, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_ocr_result(ref MDOCRResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_ocr_model(ref MDModel model);

        #endregion

        #region OCR 单行识别 OcrRecognition

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_ocr_recognition_model(ref MDModel model, string modelPath,
            string dictPath, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_recognition_model_predict(ref MDModel model, ref MDImage image,
            ref MDOCRResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_ocr_recognition_model_predict_batch(ref MDModel model, ref MDImage image,
            int batchSize, ref MDPolygon polygons, int size, ref MDOCRResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_ocr_recognition_result(ref MDOCRResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_ocr_recognition_model(ref MDModel model);

        #endregion

        #region 表格结构识别 PaddleStructureTable

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_structure_table_model(ref MDModel model,
            ref MDStructureTableModelParameters parameters, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_structure_table_model_predict(ref MDModel model, ref MDImage image,
            ref MDOCRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_structure_table_result(ref MDImage image, ref MDOCRResults cResults,
            string fontPath, int fontSize, double alpha, int saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_structure_table_result(ref MDOCRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_structure_table_result(ref MDOCRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_structure_table_model(ref MDModel model);

        #endregion

        #region 车牌检测 LprDetection

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_lpr_det_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_lpr_det_predict(ref MDModel model, ref MDImage image,
            ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_lpr_det_result(ref MDImage image, ref MDKeyPointResults cResults,
            string fontPath, int fontSize, int landmarkRadius, double alpha, int saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_lpr_det_result(ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_det_result(ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_det_model(ref MDModel model);

        #endregion

        #region 车牌识别 LprRecognizer

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_lpr_rec_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_lpr_rec_predict(ref MDModel model, ref MDImage image,
            ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_lpr_rec_result(ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_rec_result(ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_rec_model(ref MDModel model);

        #endregion

        #region 车牌检测+识别 pipeline LprPipeline

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_lpr_pipeline_model(ref MDModel model, string lprDetModelFile,
            string lprRecModelFile, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_lpr_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_lpr_pipeline_result(ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_lpr_pipeline_result(ref MDImage image, ref MDLPRResults cResults,
            string fontPath, int fontSize, int landmarkRadius, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_pipeline_result(ref MDLPRResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_lpr_pipeline_model(ref MDModel model);

        #endregion

        #region 行人属性 pipeline PedestrianAttribute

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_attr_model(
            ref MDModel model, string detModelPath, string clsModelPath, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_attr_det_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_attr_cls_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_attr_cls_batch_size(ref MDModel model, int batchSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_attr_det_threshold(ref MDModel model, float threshold);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_attr_model_predict(ref MDModel model, ref MDImage image,
            ref MDAttributeResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_attr_result(ref MDAttributeResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_attr_result(ref MDImage image, ref MDAttributeResults result,
            double threshold, ref MDMapData labelMap, string fontPath, int fontSize, double alpha, bool saveResult,
            int[] abnormalIds, int abnormalIdsSize, int showAttr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_attr_result(ref MDAttributeResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_attr_model(ref MDModel model);

        #endregion

        #region ASR SenseVoice

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_sense_voice_model(ref MDModel model,
            ref MDSenseVoiceParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_sense_voice_model_predict(ref MDModel model, string wav_path,
            ref MDASRResult asr_result, int audio_fs = 16000);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_sense_voice_result(ref MDASRResult asr_result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_sense_voice_model(ref MDModel model);

        #endregion

        #region TTS Kokoro

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_kokoro_model(ref MDModel model, ref MDKokoroParameters parameters,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_kokoro_model_predict(ref MDModel model, IntPtr text, string voice, float speed,
            ref MDTTSResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_write_wav(ref MDTTSResult result, string outputPath);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_free_kokoro_result(ref MDTTSResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_kokoro_model(ref MDModel model);

        #endregion
    }
}
