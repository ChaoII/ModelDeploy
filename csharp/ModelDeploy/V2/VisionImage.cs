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

        // NV12 零拷贝 frame 引用的托管缓冲 pin：frame 存活期内禁止 GC 移动调用方 byte[]。
        private GCHandle _pinY;
        private GCHandle _pinUv;
        private bool _hasPinY;
        private bool _hasPinUv;

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

        /// <summary>
        /// 包装 predict_nv12 输出的绑定输入帧 ImageData（设备相关的 NV12 帧，库内不属主）。
        /// 生命周期由本对象管理（Dispose 调用 md_image_destroy，仅释放包装句柄，不碰输入缓冲）。
        /// </summary>
        public static VisionImage FromDeviceFrame(IntPtr handle)
        {
            if (handle == IntPtr.Zero) throw new ArgumentException("Frame handle must be non-zero", nameof(handle));
            return new VisionImage(handle);
        }

        /// <summary>
        /// Pin 本 frame 零拷贝引用的托管缓冲（调用方 byte[] y/uv），禁止 GC 在 frame 存活期内移动它们。
        /// 仅用于持托管数组源的设备帧路径；未持托管缓冲的路径（如解码器 IntPtr 源）不调用。
        /// 由 Dispose / 析构统一 Free。
        /// </summary>
        internal void PinBuffers(byte[] y, byte[] uv)
        {
            _pinY = GCHandle.Alloc(y, GCHandleType.Pinned);
            _hasPinY = true;
            if (uv != null)
            {
                _pinUv = GCHandle.Alloc(uv, GCHandleType.Pinned);
                _hasPinUv = true;
            }
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
                if (_hasPinY) { _pinY.Free(); _hasPinY = false; }
                if (_hasPinUv) { _pinUv.Free(); _hasPinUv = false; }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~VisionImage() => Dispose();
    }
}
