//
// P/Invoke 统一入口：ModelDeploySDK (capi) 的全部 extern 集中在此管理。
// 模型类通过 `using static ModelDeploy.NativeMethods;` 直接裸调用。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    /// <summary>
    /// ModelDeploySDK 原生绑定统一入口（partial，按模块拆文件）。
    /// 全部基于 capi：不透明句柄（IntPtr）+ 数组式 getter。
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
        internal static extern MDStatus md_option_set_device(IntPtr opt, int dev, int device_id);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_backend(IntPtr opt, int backend);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_cpu_threads(IntPtr opt, int n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_fp16(IntPtr opt, int enable);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_external_stream(IntPtr opt, IntPtr s);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_password(IntPtr opt, IntPtr pwd);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_model_path(IntPtr opt, IntPtr path, IntPtr pwd);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_model_buffer(IntPtr opt, byte[] data, UIntPtr len, IntPtr fmt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_option_set_config(IntPtr opt, IntPtr ns, IntPtr key, IntPtr value);

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
            int w, int h, int step_y, int step_uv);

        // owned 版：拷入自有缓冲，产真 NV12 两平面帧（调用方无需保活）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_nv12_owned(out IntPtr handle, byte[] y, byte[] uv,
            int w, int h, int step_y, int step_uv);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_device_nv12(out IntPtr handle, byte[] y, byte[] uv,
            int w, int h, int step_y, int step_uv, int dev);

        // 设备指针零拷贝工厂（y/uv 为设备内存指针）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_from_device_nv12(out IntPtr handle, IntPtr y, IntPtr uv,
            int w, int h, int step_y, int step_uv, int dev);

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

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_info(IntPtr handle, out int type, out int dev, out int nplanes);

        // 取 NV12 帧平面指针（dev 返回帧所在设备；仅对 NV12 有效，CPU BGR 图返回 UNSUPPORTED_TYPE）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_plane_ptrs(IntPtr handle, out int dev, out IntPtr y, out IntPtr uv);

        // 裸指针设备归属校验：CPU 非空→OK；CPU 空→INVALID_ARGUMENT；GPU（未编 CUDA 时）/TPU→UNSUPPORTED_TYPE
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_ptr_validate_device(IntPtr ptr, int dev, int device_id);

        // 取整图/单平面的原生连续主机字节（buf 由库内部持有，立即 Marshal.Copy 复制）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_to_host_bytes(IntPtr h, out IntPtr buf, out UIntPtr n, out int format);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_image_plane_bytes(IntPtr h, int i, out IntPtr buf, out UIntPtr n, out int step);

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
        internal static extern MDStatus md_model_set_cls_batch_size(IntPtr handle, int batch);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_rec_batch_size(IntPtr handle, int batch);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_rec_image_shape(IntPtr handle, int c, int h, int w);

        // 模型前/后处理参数（扁平参数名，见 md_model_param_names/type）
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_param_i(IntPtr model, string name, long value);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_param_d(IntPtr model, string name, double value);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_param_b(IntPtr model, string name, int enable);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_set_param_s(IntPtr model, string name, string value);

        // 自省（kind 级，无需模型就绪）：names 以 '|' 分隔（库持有，无需释放）；type_out 为 'I'/'D'/'B'/'S'
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_param_names(int kind, out IntPtr names);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_param_type(int kind, string name, out byte typeOut);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_model_predict(IntPtr handle, IntPtr image, out IntPtr result);

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
        internal static extern MDStatus md_audio_asr_wav_result(IntPtr handle,
            string wavPath, out MDAsrResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_asr_result(IntPtr handle, float[] samples, UIntPtr n,
            int sampleRate, out MDAsrResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_tts(IntPtr handle,
IntPtr text,
IntPtr voice, float speed,
            out int sampleRate, out IntPtr audio, out UIntPtr audioN);

        // 声纹（SpeakerVerify）：提取说话人 embedding。返回的 embedding 为借用指针
        // （归模型句柄所有），调用方用 Marshal.Copy 立即复制，勿阻塞/复用。
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_speaker_embed(IntPtr handle, float[] samples, UIntPtr n,
            out IntPtr embedding, out UIntPtr embN);

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
        internal static extern MDStatus md_result_reid_embedding(IntPtr handle, UIntPtr i,
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
        internal static extern MDStatus md_result_formula(IntPtr handle, UIntPtr i, out IntPtr latex);

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

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_age_batch(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_gender_batch(IntPtr handle, out IntPtr items, out UIntPtr count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_sem_seg_batch(IntPtr handle, UIntPtr i, out IntPtr labels,
            out UIntPtr outH, out UIntPtr outW, out int numClasses);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_depth_batch(IntPtr handle, UIntPtr i, out IntPtr depth,
            out UIntPtr outH, out UIntPtr outW);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_ocr_batch_count(IntPtr handle, out UIntPtr n);

        // ==================== 2D 批量结果 getter（按图索引，逐图取项数组） ====================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_detection_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_classification_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_pose_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_keypoints_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr kps, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_obb_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_instance_seg_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_mask_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr buf,
            out UIntPtr h, out UIntPtr w);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face_kps_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr kps, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_face_embedding_batch(IntPtr handle, UIntPtr img, out IntPtr embedding, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_kps_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr kps, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_embedding_batch(IntPtr handle, UIntPtr img, UIntPtr item,
            out IntPtr embedding, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_insightface_pose_batch(IntPtr handle, UIntPtr img, UIntPtr item,
            out IntPtr pose, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_ocr_batch(IntPtr handle, UIntPtr img, UIntPtr line, out IntPtr quad,
            out IntPtr text, out float score);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_ocr_cls_batch(IntPtr handle, UIntPtr img, UIntPtr line, out int clsLabel, out float clsScore);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_lpr_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_plate_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr plate, out IntPtr color);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_lpr_keypoints_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr kps, out UIntPtr n);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_attribute_batch(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_result_attr_scores_batch(IntPtr handle, UIntPtr img, UIntPtr item, out IntPtr scores, out UIntPtr n);

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

        #region 条码识别

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_barcode_create(out IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_barcode_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_barcode_set_formats(IntPtr handle, uint formats);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_barcode_detect(IntPtr handle, IntPtr image,
            [Out] MDBarcodeItem[] items, ref uint count);

        #endregion

        #region 多目标跟踪

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_tracker_create(MDTrackerKind kind, out IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_tracker_destroy(IntPtr handle);

        // 非变异容量查询：计算 update(n) 会产生的输出数，不推进跟踪器状态。调用方先查询，
        // 再分配该容量，最后调用一次有状态 update（见 md_tracker_update 契约）。
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_tracker_capacity(IntPtr handle, MDBox[] boxes,
            float[] scores, int[] labelIds, UIntPtr n, out UIntPtr outCount);

        // 有状态更新提交（每逻辑帧恰一次）。out 容量 = 进入时 *out_count（size_t）；不足时返回
        // MD_ERR_INVALID_ARGUMENT 且 *out_count 置为需要数。调用方应以 md_tracker_capacity 先行。
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_tracker_update(IntPtr handle, MDBox[] boxes,
            float[] scores, int[] labelIds, UIntPtr n,
            [Out] MDTrackItem[] outItems, ref UIntPtr outCount);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_tracker_set_params(IntPtr handle, string name, double value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_tracker_reset(IntPtr handle);

        #endregion

        #region 解决方案（vision::solution / tool）

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_create(out IntPtr handle, MDSolutionKind kind);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_object_counter_set_line(IntPtr handle, float ax, float ay, float bx, float by);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_object_counter_update(IntPtr handle, float[] boxes, UIntPtr n, int[] labelIds, int[] trackIds);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_object_counter_hline(IntPtr handle, out int inCount, out int outCount);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_heatmap_set_size(IntPtr handle, int w, int h);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_heatmap_update(IntPtr handle, float[] boxes, UIntPtr n, int frameW, int frameH);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_heatmap_peak(IntPtr handle, out int x, out int y);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_vision_iou4(float ax, float ay, float aw, float ah,
            float bx, float by, float bw, float bh, out float iou);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_region_counter_add(IntPtr handle, string name, float[] xy, UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_region_counter_update(IntPtr handle, float[] boxes, UIntPtr n, int[] trackIds, int[] labelIds);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_solution_region_counter_count(IntPtr handle, string name);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_queue_set_region(IntPtr handle, float[] xy, UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_queue_update(IntPtr handle, float[] boxes, UIntPtr n, int[] trackIds, int[] labelIds);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_solution_queue_count(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_track_zone_set_region(IntPtr handle, float[] xy, UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_solution_track_zone_update(IntPtr handle, float[] boxes, UIntPtr n, int[] trackIds, int[] labelIds);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_solution_track_zone_count(IntPtr handle);

        #endregion

        #region 音频解决方案（audio::solution / tool）

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_solution_create(out IntPtr handle, MDAudioSolutionKind kind);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_solution_destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_speaker_search_enroll(IntPtr handle, IntPtr label, float[] emb, UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_speaker_search_match(IntPtr handle, float[] emb, UIntPtr n, int k,
            out IntPtr bestLabel, out float bestScore);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_audio_resample(float[] input, UIntPtr n, int inSr, int outSr,
            out IntPtr output, out UIntPtr outN);

        #endregion

        #region NLP 工具 / TextClassifier

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_nlp_split_sent(IntPtr text, out IntPtr sents, out UIntPtr n);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_nlp_stats(IntPtr text, out UIntPtr chars, out UIntPtr words, out UIntPtr sents);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDStatus md_nlp_classify(IntPtr handle, IntPtr text, out int label, out float score);

        #endregion
    }
}
