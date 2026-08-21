using System;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy
{
    /// <summary>capi 绘制工具（就地绘制到 VisionImage）。</summary>
    public static class Draw
    {
        public static void DrawRect(VisionImage image, float x, float y, float w, float h,
            byte r, byte g, byte b, float alpha = 0.5f)
        {
            var c = new MDColorRGBA { r = r, g = g, b = b, a = 255 };
            var status = md_draw_rect(image.Handle, x, y, w, h, c, alpha);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Draw rect failed: {BaseModel.GetLastError()}");
        }

        public static void DrawPolygon(VisionImage image, float[] xs, float[] ys,
            byte r, byte g, byte b, float alpha = 0.5f)
        {
            if (xs.Length != ys.Length || xs.Length < 3)
                throw new ArgumentException("xs/ys must be same length >= 3");
            var c = new MDColorRGBA { r = r, g = g, b = b, a = 255 };
            var status = md_draw_polygon(image.Handle, xs, ys, new UIntPtr((uint)xs.Length), c, alpha);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Draw polygon failed: {BaseModel.GetLastError()}");
        }

        public static void DrawText(VisionImage image, float x, float y, string text,
            string fontPath, int fontSize, byte r, byte g, byte b, float alpha = 1.0f)
        {
            var c = new MDColorRGBA { r = r, g = g, b = b, a = 255 };
            var status = md_draw_text(image.Handle, x, y, text, fontPath ?? string.Empty, fontSize, c, alpha);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Draw text failed: {BaseModel.GetLastError()}");
        }
    }
}
