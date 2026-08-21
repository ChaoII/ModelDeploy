using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy
{
    /// <summary>
    /// 预测结果：同时持有原生结果句柄 + 惰性读取的托管列表。
    ///  - 纯推理：直接 foreach / 索引读取（实现 IReadOnlyList&lt;T&gt;）。
    ///  - 可视化：调用 Draw(canvas, options)，句柄直达 C++ vis_*（零重建、最忠实）。
    ///  - 生命周期：using / Dispose 释放结果句柄。
    /// </summary>
    public sealed class Prediction<T> : IReadOnlyList<T>, IDisposable
    {
        private readonly IntPtr _result;
        private readonly Func<IntPtr, T[]> _reader;
        private T[] _cache;
        private bool _disposed;

        internal Prediction(IntPtr result, Func<IntPtr, T[]> reader)
        {
            _result = result;
            _reader = reader;
        }

        /// <summary>结果句柄（高级用法）。</summary>
        public IntPtr Handle => _result;

        private T[] Items => _cache ??= _reader(_result);

        public int Count => Items.Length;
        public T this[int index] => Items[index];
        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)Items).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// 把结果就地绘制到 canvas（复用 C++ vis_*，句柄直达原始结果）。
        /// 可选操作：不需要可视化时不调用，零开销。
        /// </summary>
        public void Draw(VisionImage canvas, DrawOptions options)
        {
            if (_result == IntPtr.Zero) throw new ObjectDisposedException(nameof(Prediction<T>));
            if (canvas == null) throw new ArgumentNullException(nameof(canvas));
            if (options == null) throw new ArgumentNullException(nameof(options));

            var native = options.ToNative();
            try
            {
                var status = md_draw_result(canvas.Handle, _result, ref native);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"Draw result failed: {BaseModel.GetLastError()}");
            }
            finally
            {
                DrawOptions.FreeNative(native);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_result != IntPtr.Zero)
                {
                    md_result_destroy(_result);
                    _cache = null;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~Prediction() => Dispose();
    }
}
