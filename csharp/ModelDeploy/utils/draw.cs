using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.utils
{
    public static class Draw
    {
        /// <summary>
        /// 在图像上绘制文本。
        /// </summary>
        public static void DrawText(Image image, Rect rect, string text, string fontPath, int fontSize, Color color, double alpha)
        {
            var cRect = rect.ToNative();
            md_draw_text(ref image.RawImage, ref cRect, text, fontPath, fontSize, color.ToNative(), alpha);
            image.Update();
        }

        /// <summary>
        /// 在图像上绘制矩形。
        /// </summary>
        public static void DrawRect(Image image, Rect rect, Color color, double alpha)
        {
            md_draw_rect(ref image.RawImage, rect.ToNative(), color.ToNative(), alpha);
            image.Update();
        }

        /// <summary>
        /// 在图像上绘制多边形。
        /// </summary>
        public static void DrawPolygon(Image image, Polygon polygon, Color color, double alpha)
        {
            var cPolygon = polygon.CopyToMDPolygon();
            try
            {
                md_draw_polygon(ref image.RawImage, cPolygon, color.ToNative(), alpha);
            }
            finally
            {
                
                Marshal.FreeHGlobal(cPolygon.data); // 保证内存释放
            }
            image.Update();
        }

        #region DllImports

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_text(ref MDImage image, ref MDRect rect, string text, string fontPath,
            int fontSize, MDColor color, double alpha);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_rect(ref MDImage image, MDRect rect, MDColor color, double alpha);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_polygon(ref MDImage image, MDPolygon polygon, MDColor color, double alpha);

        #endregion
    }
}
