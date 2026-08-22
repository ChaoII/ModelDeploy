using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy
{
    /// <summary>
    /// 单个条码 / 二维码解码结果（对应 capi MD_BarcodeItem）。
    /// </summary>
    public sealed class BarcodeResult
    {
        public string Text { get; set; }
        public string Format { get; set; }
        public PointF[] Quad { get; set; }
        public float Score { get; set; }
        public bool IsQr { get; set; }

        public override string ToString() =>
            $"{Format}[{Text}] score={Score:F3} is_qr={IsQr}";
    }

    /// <summary>
    /// 条码 / 二维码识别器（纯 CV，无模型依赖；基于 ZXing 解码）。
    /// </summary>
    public sealed class BarcodeDetector : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public BarcodeDetector()
        {
            if (md_barcode_create(out _handle) != MDStatus.MD_OK)
                throw new InvalidOperationException($"barcode create failed: {BaseModel.GetLastError()}");
        }

        /// <summary>限定解码格式子集（FMT_* 位或，0 = 全部）。</summary>
        public void SetFormats(uint formats)
        {
            EnsureNotDisposed();
            if (md_barcode_set_formats(_handle, formats) != MDStatus.MD_OK)
                throw new InvalidOperationException($"set_formats failed: {BaseModel.GetLastError()}");
        }

        /// <summary>
        /// 检测并解码图片中的所有码。遵循 capi 查询/提交契约：
        /// 先以 items=null 做容量查询，分配缓冲后再正式读取一次。
        /// </summary>
        public BarcodeResult[] Detect(VisionImage image)
        {
            EnsureNotDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            uint need = 0;
            md_barcode_detect(_handle, image.Handle, null, ref need);
            if (need == 0) return Array.Empty<BarcodeResult>();
            var items = new MDBarcodeItem[need];
            uint written = need;
            if (md_barcode_detect(_handle, image.Handle, items, ref written) != MDStatus.MD_OK)
                throw new InvalidOperationException($"detect failed: {BaseModel.GetLastError()}");
            var result = new BarcodeResult[written];
            for (int i = 0; i < (int)written; i++)
            {
                var it = items[i];
                result[i] = new BarcodeResult
                {
                    Text = it.Text,
                    Format = it.Format,
                    Quad = new[]
                    {
                        new PointF(it.Quad0_x, it.Quad0_y),
                        new PointF(it.Quad1_x, it.Quad1_y),
                        new PointF(it.Quad2_x, it.Quad2_y),
                        new PointF(it.Quad3_x, it.Quad3_y)
                    },
                    Score = it.Score,
                    IsQr = it.IsQr != 0
                };
            }
            return result;
        }

        private void EnsureNotDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BarcodeDetector));
        }

        public void Dispose()
        {
            if (_disposed) return;
            if (_handle != IntPtr.Zero) md_barcode_destroy(_handle);
            _handle = IntPtr.Zero;
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~BarcodeDetector() { Dispose(); }
    }
}
