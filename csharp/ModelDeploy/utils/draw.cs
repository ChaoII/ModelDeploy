using ModelDeploy.types_internal_c;
using System.Runtime.InteropServices;

namespace ModelDeploy.utils
{
    public class Draw
    {
        public static void DrawText(Image image, MDRect rect, string text, string fontPath, int fontSize, MDColor color,
            double alpha)
        {
            md_draw_text(ref image.RawImage, ref rect, text, fontPath, fontSize, color, alpha);
            image.Update();
        }

        public static void DrawRect(Image image, MDRect rect, MDColor color, double alpha)
        {
            md_draw_rect(ref image.RawImage, rect, color, alpha);
            image.Update();
        }

        public static void DrawPolygon(Image image, MDPolygon polygon, MDColor color, double alpha)
        {
            md_draw_polygon(ref image.RawImage, polygon, color, alpha);
            image.Update();
        }

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_text(ref MDImage image, ref MDRect rect, string text, string fontPath,
            int fontSize, MDColor color, double alpha);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_rect(ref MDImage image, MDRect rect, MDColor color, double alpha);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void md_draw_polygon(ref MDImage image, MDPolygon polygon, MDColor color, double alpha);
    }
}
