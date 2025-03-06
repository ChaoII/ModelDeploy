using ModelDeploy.types_internal_c;
using System.Runtime.InteropServices;

namespace ModelDeploy.utils
{
    public class Draw
    {
        public static void DrawText(Image image, Rect rect, string text, string fontPath, int fontSize, Color color,
            double alpha)
        {
            MDRect cRect = rect.ToRaw();
            md_draw_text(ref image.RawImage, ref cRect, text, fontPath, fontSize, color.ToRaw(), alpha);
            image.Update();
        }

        public static void DrawRect(Image image, Rect rect, Color color, double alpha)
        {
            md_draw_rect(ref image.RawImage, rect.ToRaw(), color.ToRaw(), alpha);
            image.Update();
        }

        public static void DrawPolygon(Image image, Polygon polygon, Color color, double alpha)
        {
            MDPolygon cPolygon = polygon.CopyToMDPolygon();
            md_draw_polygon(ref image.RawImage, cPolygon, color.ToRaw(), alpha);
            Marshal.FreeHGlobal(cPolygon.data);
            image.Update();
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_text(ref MDImage image, ref MDRect rect, string text, string fontPath,
            int fontSize, MDColor color, double alpha);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_rect(ref MDImage image, MDRect rect, MDColor color, double alpha);

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_polygon(ref MDImage image, MDPolygon polygon, MDColor color, double alpha);
    }
}