//
// 视觉模型 P/Invoke：检测 / 分类 / 关键点(pose) / OBB / 实例分割 / 语义分割 / 深度估计。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    internal static partial class NativeMethods
    {
        #region 目标检测 UltralyticsDet

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_detection_model(ref MDModel model, string modelDir,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_detection_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_detection_predict(ref MDModel model, ref MDImage image,
            ref MDDetectionResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_detection_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDDetectionResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_detection_result(ref MDDetectionResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_detection_result(ref MDImage image, ref MDDetectionResults result,
            double threshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_detection_result(ref MDDetectionResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_detection_model(ref MDModel model);

        #endregion

        #region 图像分类 UltralyticsCls

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_classification_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_classification_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_disable_classification_center_crop(ref MDModel model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_classification_multi_label(ref MDModel model, int isMutiLabel);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_classification_predict(ref MDModel model, ref MDImage image,
            ref MDClassificationResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_classification_result(ref MDImage image,
            ref MDClassificationResults result,
            int topK, float scoreThreshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_classification_result(ref MDClassificationResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_classification_model(ref MDModel model);

        #endregion

        #region 关键点 / 姿态 UltralyticsPose

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_keypoint_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_keypoint_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_keypoint_predict(ref MDModel model, ref MDImage image,
            ref MDKeyPointResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_pose_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDKeyPointResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_keypoint_result(ref MDKeyPointResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_keypoint_result(ref MDImage image, ref MDKeyPointResults result,
            string fontPath, int fontSize, int keypointRadius, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_keypoint_result(ref MDKeyPointResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_keypoint_model(ref MDModel model);

        #endregion

        #region OBB UltralyticsObb

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_obb_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_obb_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_obb_predict(ref MDModel model, ref MDImage image,
            ref MDObbResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_obb_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDObbResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_obb_result(ref MDObbResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_obb_result(ref MDImage image, ref MDObbResults result,
            double threshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_obb_result(ref MDObbResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_obb_model(ref MDModel model);

        #endregion

        #region 实例分割 UltralyticsSeg(iseg)

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            md_create_instance_seg_model(ref MDModel model, string modelPath, ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_instance_seg_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_instance_seg_predict(ref MDModel model, ref MDImage image,
            ref MDISegResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_iseg_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDISegResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_print_instance_seg_result(ref MDISegResults cResults);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_instance_seg_result(ref MDImage image, ref MDISegResults result,
            double threshold, string fontPath, int fontSize, double alpha, bool saveResult);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_instance_seg_result(ref MDISegResults results);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_instance_seg_model(ref MDModel model);

        #endregion

        #region 语义分割 UltralyticsSem

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_sem_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_sem_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_sem_predict(ref MDModel model, ref MDImage image,
            ref MDSemSegResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_sem_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDSemSegResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_sem_result(ref MDSemSegResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_sem_model(ref MDModel model);

        #endregion

        #region 深度估计 UltralyticsDepth

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_create_depth_model(ref MDModel model, string modelPath,
            ref MDRuntimeOption option);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_set_depth_input_size(ref MDModel model, MDSize size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_depth_predict(ref MDModel model, ref MDImage image,
            ref MDDepthResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_depth_predict_nv12(ref MDModel model, IntPtr srcY, IntPtr srcUV,
            int width, int height, int stepY, int stepUV, MDDevice srcDevice,
            ref MDDepthResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_depth_result(ref MDDepthResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_depth_model(ref MDModel model);

        #endregion
    }
}
