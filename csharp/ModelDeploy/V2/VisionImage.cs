using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.V2
{
    /// <summary>图像像素类型（对齐 csrc/vision/common/basic_types.h 的 ImageType）。</summary>
    public enum MdImageType
    {
        GRAY_U8 = 0,
        PLA_BGR_U8 = 20, PLA_RGB_U8 = 21,
        PKG_BGR_U8 = 22, PKG_RGB_U8 = 23,
        PLA_BGRA_U8 = 24, PLA_RGBA_U8 = 25, PKG_BGRA_U8 = 26, PKG_RGBA_U8 = 27,
        NV12 = 60, NV21 = 61, I420 = 62,
        UNKNOWN = 63
    }

    /// <summary>图像平面描述（Data 指向平面像素，Step 为行距/跨度）。</summary>
    public struct Plane
    {
        public IntPtr Data;
        public int Step;
    }

    /// <summary>
    /// capi2 图像包装：内部持有 MDImageHandle（IntPtr），生命周期由本类管理。
    /// 对齐 C++ ImageData：类型/设备/平面数元信息 + 主机/NV12 设备工厂。
    /// </summary>
    public sealed class VisionImage : IDisposable
    {
        public IntPtr Handle { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }

        /// <summary>像素类型（由原生 md_image_info 读出）。</summary>
        public MdImageType Type { get; private set; }

        /// <summary>像素内存所在设备（由原生 md_image_info 读出）。</summary>
        public ModelDeploy.Device Device { get; private set; }

        /// <summary>平面数（由原生 md_image_info 读出）。</summary>
        public int PlaneCount { get; private set; }

        /// <summary>通道数由 Type 推导（BGR/RGB 三通道，灰度与 YUV 单通道，BGRA/RGBA 四通道）。</summary>
        public int Channels => Type switch
        {
            MdImageType.NV12 or MdImageType.NV21 or MdImageType.I420 => 1,
            MdImageType.GRAY_U8 => 1,
            MdImageType.PLA_BGRA_U8 or MdImageType.PLA_RGBA_U8
                or MdImageType.PKG_BGRA_U8 or MdImageType.PKG_RGBA_U8 => 4,
            _ => 3
        };

        private bool _disposed;

        // NV12 零拷贝 frame 引用的托管缓冲 pin：frame 存活期内禁止 GC 移动调用方 byte[]。
        private GCHandle _pinY;
        private GCHandle _pinUv;
        private bool _hasPinY;
        private bool _hasPinUv;

        // NV12 构造时记录的平面行距，供 GetPlane 返回 Step。
        private int _stepY;
        private int _stepUv;

        private VisionImage(IntPtr handle, int stepY = 0, int stepUv = 0)
        {
            Handle = handle;
            _stepY = stepY;
            _stepUv = stepUv;
            md_image_size(handle, out var w, out var h);
            Width = w;
            Height = h;
            md_image_info(handle, out int type, out int dev, out int nplanes);
            Type = (MdImageType)type;
            Device = (ModelDeploy.Device)dev;
            PlaneCount = nplanes;
        }

        /// <summary>返回第 i 个平面（仅 NV12/NV21 支持）。</summary>
        public Plane GetPlane(int i)
        {
            if (Type != MdImageType.NV12 && Type != MdImageType.NV21)
                throw new NotSupportedException($"GetPlane: only NV12/NV21 supported (type={Type})");
            md_image_plane_ptrs(Handle, out int dev, out IntPtr y, out IntPtr uv);
            if (i == 0) return new Plane { Data = y, Step = _stepY > 0 ? _stepY : Width };
            if (i == 1) return new Plane { Data = uv, Step = _stepUv > 0 ? _stepUv : Width };
            throw new ArgumentOutOfRangeException(nameof(i));
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

        /// <summary>
        /// 用外部 y/uv 设备指针零拷贝构造 NV12 设备帧（库不拥有内存，调用方保证指针存活到预测结束）。不 pin。
        /// </summary>
        public static VisionImage FromDeviceNv12(IntPtr y, IntPtr uv, int w, int h, int stepY = 0, int stepUv = 0, ModelDeploy.Device dev = ModelDeploy.Device.CPU)
        {
            if (y == IntPtr.Zero) throw new ArgumentNullException(nameof(y));
            if (uv == IntPtr.Zero) throw new ArgumentNullException(nameof(uv));
            var status = md_image_from_device_nv12(out var hh, y, uv, w, h, stepY, stepUv, (int)dev);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"FromDeviceNv12 failed: {BaseModel.GetLastError()}");
            return new VisionImage(hh, stepY, stepUv);
        }

        /// <summary>
        /// 由托管 Y/UV 缓冲构造真正的 NV12 2 平面 CPU 图：内置 pin（_pinY/_pinUv）禁止 GC 移动缓冲，
        /// 以 pin 指针调设备工厂（dev=CPU），等价 C++ from_planes(..., NV12, Device::CPU)。
        /// 缓冲在 Dispose 时释放。
        /// </summary>
        public static VisionImage FromNv12Data(byte[] y, byte[] uv, int w, int h, int stepY = 0, int stepUv = 0)
        {
            if (y == null) throw new ArgumentNullException(nameof(y));
            if (uv == null) throw new ArgumentNullException(nameof(uv));
            var pinY = GCHandle.Alloc(y, GCHandleType.Pinned);
            var pinUv = GCHandle.Alloc(uv, GCHandleType.Pinned);
            try
            {
                var status = md_image_from_device_nv12(out var hh, pinY.AddrOfPinnedObject(), pinUv.AddrOfPinnedObject(),
                    w, h, stepY, stepUv, (int)ModelDeploy.Device.CPU);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"Image from NV12 failed: {BaseModel.GetLastError()}");
                var v = new VisionImage(hh, stepY, stepUv);
                v._pinY = pinY;
                v._hasPinY = true;
                v._pinUv = pinUv;
                v._hasPinUv = true;
                return v;
            }
            catch
            {
                pinY.Free();
                pinUv.Free();
                throw;
            }
        }

        /// <summary>
        /// 包装 md_image_from_device_nv12 构造的绑定输入帧 ImageData（设备相关的 NV12 帧，库内不属主）。
        /// 内部供 BaseModel NV12 直接输入路径使用；Dispose 仅释放包装句柄，不碰输入缓冲。
        /// </summary>
        internal static VisionImage FromDeviceFrame(IntPtr handle)
        {
            if (handle == IntPtr.Zero) throw new ArgumentException("Frame handle must be non-zero", nameof(handle));
            return new VisionImage(handle);
        }

        /// <summary>
        /// Pin 本 frame 零拷贝引用的托管缓冲（调用方 byte[] y/uv），禁止 GC 在 frame 存活期内移动它们。
        /// 内部供 BaseModel NV12 直接输入路径使用；由 Dispose / 析构统一 Free。
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
