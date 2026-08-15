//
// 人脸模型 P/Invoke：SCRFD / 人脸识别 / 年龄 / 性别 / 活体 / 人脸识别 pipeline / insightface。
//

using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    internal static partial class NativeMethods
    {
        #region SCRFD 人脸检测 Scrfd

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_face_det_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_det_predict(ref MDModel model, ref MDImage image,
            ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_face_det_result(ref MDImage image,
            ref MDKeyPointResults cResults, string fontPath, int fontSize,
            int landmarkRadius, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_det_result(ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_det_model(ref MDModel model);

        #endregion

        #region 人脸识别 SeetaFaceId

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_face_rec_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_rec_predict(ref MDModel model, ref MDImage image,
            ref MDFaceRecognizerResult cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_face_rec_result(ref MDFaceRecognizerResult cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_rec_result(ref MDFaceRecognizerResult cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_rec_model(ref MDModel model);

        #endregion

        #region 年龄 SeetaFaceAge

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_face_age_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_age_predict(ref MDModel model, ref MDImage image,
            ref int cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_age_model(ref MDModel model);

        #endregion

        #region 性别 SeetaFaceGender

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_face_gender_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_gender_predict(ref MDModel model, ref MDImage image,
            ref int cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_gender_model(ref MDModel model);

        #endregion

        #region 活体检测 SeetaFaceAsFirst / SeetaFaceAsSecond / SeetaFaceAntiSpoof

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_face_as_first_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_as_first_predict(ref MDModel model, ref MDImage image, ref float cResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_as_first_model(ref MDModel model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            md_create_face_as_second_model(ref MDModel model, string modelPath, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_as_second_predict(ref MDModel model, ref MDImage image,
            ref MDFaceAsSecondResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_as_second_result(ref MDFaceAsSecondResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_as_second_model(ref MDModel model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            md_create_face_as_pipeline_model(ref MDModel model, string faceDetModelFile, string firstModelFile,
                string secondModelFile, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_as_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDFaceAsResults cResults, float fuseThreshold = 0.8f, float clarityThreshold = 0.3f);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_as_pipeline_result(ref MDFaceAsResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_as_pipeline_model(ref MDModel model);

        #endregion

        #region 人脸识别 pipeline FaceRecognizerPipeline

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            md_create_face_rec_pipeline_model(ref MDModel model, string faceDetModelFile, string faceRecModelFile,
                ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_face_rec_pipeline_predict(ref MDModel model, ref MDImage image,
            ref MDFaceRecognizerResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_face_rec_pipeline_result(ref MDFaceRecognizerResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_rec_pipeline_result(ref MDFaceRecognizerResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_face_rec_pipeline_model(ref MDModel model);

        #endregion

        #region insightface 人脸分析

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_insightface_model(ref MDModel model,
            string detModelPath, string recModelPath, string lmk2dModelPath, string lmk3dModelPath,
            string genderageModelPath, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_insightface_analyze(ref MDModel model, ref MDImage image,
            ref MDInsightFaceResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_insightface_set_det_thresh(ref MDModel model, float thresh);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_insightface_result(ref MDInsightFaceResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_insightface_model(ref MDModel model);

        #endregion
    }
}
