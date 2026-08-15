// 
// P/Invoke 统一入口：所有 ModelDeploySDK 的 extern 集中在此管理。
// 模型类通过 `using static ModelDeploy.NativeMethods;` 直接裸调用。
//

using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy
{
    /// <summary>
    /// ModelDeploySDK 原生绑定统一入口（partial，按模块拆文件）。
    /// </summary>
    internal static partial class NativeMethods
    {
        private const string DllName = "ModelDeploySDK";

        #region 通用（所有模型共用）

        /// <summary>克隆模型（所有模型类共用，仅声明一次）</summary>
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int md_clone_model(ref MDModel model, ref MDModel from);

        #endregion

        #region 图像 I/O

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_read_image(string imagePath);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_compressed_bytes(byte[] data, int byteSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_to_compressed_bytes(ref MDImage image, string ext);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_bgr24_data(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_rgb24_data(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_rgb24_data_to_bgr24(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_yuv420p_data_to_bgr24(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_nv12_data_to_bgr24(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_nv21_data_to_bgr24(byte[] data, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_from_base64_str(string base64String);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_image(ref MDImage img);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_clone_image(ref MDImage image);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDImage md_crop_image(ref MDImage image, MDRect rect);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_show_image(ref MDImage image);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_save_image(ref MDImage image, string imagePath);

        #endregion

        #region 工具

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern MDKeyValuePair md_create_key_value_pair(int key, IntPtr value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_free_md_map(ref MDMapData cMap);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool md_get_button_enable_status(ref MDImage image, int pixThreshold,
            double rateThreshold);

        #endregion

        #region 绘图

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_text(ref MDImage image, ref MDRect rect, string text, string fontPath,
            int fontSize, MDColor color, double alpha);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_rect(ref MDImage image, MDRect rect, MDColor color, double alpha);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void md_draw_polygon(ref MDImage image, MDPolygon polygon, MDColor color, double alpha);

        #endregion
    }
}
