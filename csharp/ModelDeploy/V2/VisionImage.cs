using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.V2
{
    /// <summary>
    /// capi2 图像包装：内部持有 MDImageHandle（IntPtr），生命周期由本类管理。
    /// </summary>
    public sealed class VisionImage : IDisposable
    {
        public IntPtr Handle { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Channels => 3;

        private bool _disposed;

        private VisionImage(IntPtr handle)
        {
            Handle = handle;
            md_image_size(handle, out var w, out var h);
            Width = w;
            Height = h;
        }

        public static VisionImage Read(string path)
        {
            var status = md_image_from_file(out var h, path);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image load failed: {BaseModel.GetLastError()}");
            return new VisionImage(h);
        }

        public static VisionImage FromBgr24Data(byte[] data, int w, int h)
        {
            var status = md_image_from_bgr24(out var hh, data, w, h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image from BGR failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public static VisionImage FromRgb24Data(byte[] data, int w, int h)
        {
            var status = md_image_from_rgb24(out var hh, data, w, h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image from RGB failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public static VisionImage FromNv12Data(byte[] y, byte[] uv, int w, int h, int stepY = 0, int stepUv = 0)
        {
            var status = md_image_from_nv12(out var hh, y, uv, w, h, stepY, stepUv, (int)Device.CPU);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image from NV12 failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public static VisionImage FromYuv420PData(byte[] data, int w, int h)
        {
            var status = md_image_from_yuv420p(out var hh, data, w, h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image from YUV420P failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public static VisionImage FromEncoded(byte[] bytes)
        {
            var status = md_image_from_encoded(out var hh, bytes, new UIntPtr((uint)bytes.Length));
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image decode failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public static VisionImage FromBase64(string b64)
        {
            var status = md_image_from_base64(out var hh, b64);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image base64 decode failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh);
        }

        public VisionImage Clone()
        {
            var status = md_image_clone(Handle, out var nh);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image clone failed: {BaseModel.GetLastError()}");
            return new VisionImage(nh);
        }

        public VisionImage Crop(int x, int y, int w, int h)
        {
            var status = md_image_crop(Handle, x, y, w, h, out var nh);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image crop failed: {BaseModel.GetLastError()}");
            return new VisionImage(nh);
        }

        public void Show()
        {
            var status = md_image_show(Handle);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image show failed: {BaseModel.GetLastError()}");
        }

        public void Save(string path)
        {
            var status = md_image_save(Handle, path);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image save failed: {BaseModel.GetLastError()}");
        }

        public byte[] Encode(string ext)
        {
            var status = md_image_encode(Handle, ext, out var buf, out var n);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Image encode failed: {BaseModel.GetLastError()}");
            return ResultReader.ReadBytes(buf, n);
        }

        /// <summary>返回 BGR 字节数组（引用图像内部数据，需在本对象存活期间使用）。</summary>
        public byte[] ToByteArray()
        {
            // BGR 数据 = Handle 内部 data；通过 encode 到 PNG 再解码不可行，直接取内部指针需要额外 API。
            // capi2 未暴露裸指针读取，走 encode(".bmp") 保证无损返回像素。
            // 说明：BMP 编码有 54 字节头，此处返回原始像素区。
            var encoded = Encode(".bmp");
            int headerSize = 54;
            int pixelBytes = Width * Height * 3;
            var pixels = new byte[pixelBytes];
            if (encoded.Length >= headerSize + pixelBytes)
                Array.Copy(encoded, headerSize, pixels, 0, pixelBytes);
            return pixels;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (Handle != IntPtr.Zero)
                {
                    md_image_destroy(Handle);
                    Handle = IntPtr.Zero;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VisionImage() => Dispose();
    }
}
