//
// P/Invoke 统一入口：ModelDeploySDK (capi2) 的全部 extern 集中在此管理。
// 模型类通过 `using static ModelDeploy.NativeMethods;` 直接裸调用。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    /// <summary>
    /// ModelDeploySDK 原生绑定统一入口（partial，按模块拆文件）。
    /// 全部基于 capi2：不透明句柄（IntPtr）+ 数组式 getter。
    /// </summary>
    internal static partial class NativeMethods
    {
        private const string DllName = "ModelDeploySDK";

        #region 错误

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr md_get_last_error();

        #endregion

        #region 运行时选项

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_create(out IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_set_device(IntPtr handle, int device);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_set_backend(IntPtr handle, int backend);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_set_cpu_threads(IntPtr handle, int n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_set_fp16(IntPtr handle, int enable);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_option_set_trt_engine_path(IntPtr handle,
string path);

        #endregion

        #region 图像

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_file(out IntPtr handle,
string path);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_bgr24(out IntPtr handle, byte[] data, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_rgb24(out IntPtr handle, byte[] data, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_nv12(out IntPtr handle, byte[] y, byte[] uv,
            int w, int h, int step_y, int step_uv, int src);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_yuv420p(out IntPtr handle, byte[] data, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_encoded(out IntPtr handle, byte[] bytes, UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_base64(out IntPtr handle,
string b64);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_clone(IntPtr inHandle, out IntPtr outHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_crop(IntPtr inHandle, int x, int y, int w, int h,
            out IntPtr outHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_show(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_save(IntPtr handle,
string path);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_encode(IntPtr handle,
string ext, out IntPtr buf, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_image_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_size(IntPtr handle, out int w, out int h);

        // 取 NV12 帧平面指针（dev 返回帧所在设备；仅对 NV12 有效，CPU BGR 图返回 UNSUPPORTED_TYPE）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_plane_ptrs(IntPtr handle, out int dev, out IntPtr y, out IntPtr uv);

        #endregion

        #region 模型

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_create(out IntPtr handle, MDModelKind kind,
string modelPath, IntPtr opt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_model_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_clone(IntPtr handle, out IntPtr outHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_ready(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_input_size(IntPtr handle, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_cls_input_size(IntPtr handle, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_predict(IntPtr handle, IntPtr image, out IntPtr result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_predict_nv12(IntPtr handle, byte[] y, byte[] uv,
            int w, int h, int stepY, int stepUv, int srcDevice, out IntPtr frame, out IntPtr result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_predict_batch(IntPtr handle, IntPtr[] imgs, UIntPtr n,
            out IntPtr result);

        #endregion

        #region 音频

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_asr_wav(IntPtr handle,
string wavPath, out IntPtr text);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_asr(IntPtr handle, float[] samples, UIntPtr n, int sampleRate,
            out IntPtr text);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_tts(IntPtr handle,
string text,
string voice, float speed,
            out int sampleRate, out IntPtr audio, out UIntPtr audioN);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_wav_save(float[] samples, UIntPtr n, int sampleRate,
string path);

        #endregion

        #region 结果

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_result_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_kind(IntPtr handle, out MDResultKind kind);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_count(IntPtr handle, out UIntPtr count);

        // 数组式 getter：一次取回 blittable 项数组 + 长度
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_detection(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_classification(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_pose(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_keypoints(IntPtr handle, UIntPtr i, out IntPtr kps, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_obb(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_instance_seg(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_mask(IntPtr handle, UIntPtr i, out IntPtr buf,
            out UIntPtr h, out UIntPtr w);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_sem_seg(IntPtr handle, out IntPtr labels,
            out UIntPtr h, out UIntPtr w, out int numClasses);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_depth(IntPtr handle, out IntPtr depth,
            out UIntPtr h, out UIntPtr w);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face_kps(IntPtr handle, UIntPtr i, out IntPtr kps, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face_embedding(IntPtr handle, UIntPtr i,
            out IntPtr embedding, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_kps(IntPtr handle, UIntPtr i,
            out IntPtr kps, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_embedding(IntPtr handle, UIntPtr i,
            out IntPtr embedding, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_pose(IntPtr handle, UIntPtr i,
            out IntPtr pose, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_ocr(IntPtr handle, UIntPtr i, out IntPtr quad,
            out IntPtr text, out float score);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_lpr(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_plate(IntPtr handle, UIntPtr i, out IntPtr plate, out IntPtr color);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_ocr_cls(IntPtr handle, UIntPtr i, out int clsLabel, out float clsScore);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_lpr_keypoints(IntPtr handle, UIntPtr i,
            out IntPtr kps, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_attribute(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_attr_scores(IntPtr handle, UIntPtr i,
            out IntPtr scores, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_age(IntPtr handle, out int age);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_gender(IntPtr handle, out int gender);

        #endregion

        #region 绘制

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_draw_rect(IntPtr handle, float x, float y, float w, float h,
            MDColorRGBA color, float alpha);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_draw_polygon(IntPtr handle, float[] xs, float[] ys, UIntPtr n,
            MDColorRGBA color, float alpha);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_draw_text(IntPtr handle, float x, float y,
            string text,
            string fontPath, int fontSize,
            MDColorRGBA color, float alpha);

        #endregion

        #region 结果可视化（复用 C++ vis_* 系列）

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_draw_result(IntPtr img, IntPtr result,
            ref MDDrawOptions options);

        #endregion
    }
}
