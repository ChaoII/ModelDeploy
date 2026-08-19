using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.V2
{
    /// <summary>
    /// capi2 运行时选项（链式 setter，替代 v1 的 MDRuntimeOption 结构体直译）。
    /// </summary>
    public sealed class RuntimeOption2
    {
        internal IntPtr Handle { get; private set; }
        private bool _ownsHandle;

        /// <summary>创建默认选项（CPU + ORT）。</summary>
        public RuntimeOption2()
        {
            md_option_create(out var h);
            Handle = h;
            _ownsHandle = true;
        }

        /// <summary>从已存在的原生句柄包装（内部使用）。</summary>
        internal RuntimeOption2(IntPtr existing, bool owns = false)
        {
            Handle = existing;
            _ownsHandle = owns;
        }

        public RuntimeOption2 UseOrt() { md_option_set_backend(Handle, 0); return this; }
        public RuntimeOption2 UseMnn() { md_option_set_backend(Handle, 1); return this; }
        public RuntimeOption2 UseTrt() { md_option_set_backend(Handle, 2); return this; }
        public RuntimeOption2 UseSophgo() { md_option_set_backend(Handle, 3); return this; }

        public RuntimeOption2 SetDevice(Device d) { md_option_set_device(Handle, (int)d); return this; }
        public RuntimeOption2 SetCpuThreads(int n) { md_option_set_cpu_threads(Handle, n); return this; }
        public RuntimeOption2 SetFp16(bool enable) { md_option_set_fp16(Handle, enable ? 1 : 0); return this; }
        public RuntimeOption2 SetTrtEnginePath(string path) { md_option_set_trt_engine_path(Handle, path); return this; }

        public void Dispose()
        {
            if (_ownsHandle && Handle != IntPtr.Zero)
            {
                md_option_destroy(Handle);
                Handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// capi2 模型基类：统一句柄生命周期 + Create/Ready/Dispose。
    /// 派生类提供强类型 Predict。
    /// </summary>
    public abstract class BaseModel : IDisposable
    {
        protected IntPtr _handle;
        protected MDModelKind _kind;
        private bool _disposed;

        protected BaseModel(MDModelKind kind, string modelPath, RuntimeOption2 opt = null)
        {
            _kind = kind;
            RuntimeOption2 ownedOpt = null;
            try
            {
                ownedOpt = opt ?? new RuntimeOption2();
                var status = md_model_create(out _handle, kind, modelPath, ownedOpt.Handle);
                if (status != MDStatus.MD_OK)
                    throw new InvalidOperationException($"Model create failed: {GetLastError()}");
            }
            finally
            {
                ownedOpt?.Dispose();
            }
        }

        public bool IsReady()
        {
            return md_model_ready(_handle) == MDStatus.MD_OK;
        }

        /// <summary>设置模型输入尺寸（pipeline 模型设置检测子模型尺寸）。</summary>
        public void SetInputSize(int w, int h)
        {
            var status = md_model_set_input_size(_handle, w, h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Set input size failed: {GetLastError()}");
        }

        /// <summary>设置 pipeline 分类子模型输入尺寸（PedestrianAttribute）。</summary>
        public void SetClsInputSize(int w, int h)
        {
            var status = md_model_set_cls_input_size(_handle, w, h);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Set cls input size failed: {GetLastError()}");
        }

        /// <summary>设置 pipeline 分类子模型 batch 大小（&gt;0 固定，-1 自动；PedestrianAttribute/OCR）。
        /// Sophgo int8 bmodel 为 batch=1 静态形状时须设为 1。</summary>
        public void SetClsBatchSize(int batch)
        {
            var status = md_model_set_cls_batch_size(_handle, batch);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Set cls batch size failed: {GetLastError()}");
        }

        /// <summary>设置 OCR 识别子模型 batch 大小（&gt;0 固定，-1 自动）。</summary>
        public void SetRecBatchSize(int batch)
        {
            var status = md_model_set_rec_batch_size(_handle, batch);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Set rec batch size failed: {GetLastError()}");
        }

        /// <summary>设置 OCR 识别子模型输入形状 (c,h,w)。</summary>
        public void SetRecImageShape(int c, int h, int w)
        {
            var status = md_model_set_rec_image_shape(_handle, c, h, w);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Set rec image shape failed: {GetLastError()}");
        }

        /// <summary>设置模型前/后处理参数（扁平参数名，见 ParamNames）。类型按重载匹配。</summary>
        public void SetParam(string name, long value)
        {
            var status = md_model_set_param_i(_handle, name, value);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"SetParam[{name}] failed: {GetLastError()}");
        }
        public void SetParam(string name, double value)
        {
            var status = md_model_set_param_d(_handle, name, value);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"SetParam[{name}] failed: {GetLastError()}");
        }
        public void SetParam(string name, bool value)
        {
            var status = md_model_set_param_b(_handle, name, value ? 1 : 0);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"SetParam[{name}] failed: {GetLastError()}");
        }
        public void SetParam(string name, string value)
        {
            var status = md_model_set_param_s(_handle, name, value);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"SetParam[{name}] failed: {GetLastError()}");
        }

        /// <summary>该模型 kind 支持的参数名（'|' 分隔）。无需模型已就绪，kind 级自省。</summary>
        public string[] ParamNames()
        {
            var status = md_model_param_names((int)_kind, out var p);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"ParamNames failed: {GetLastError()}");
            var s = p == IntPtr.Zero ? string.Empty : Utf8Helper.Read(p);
            return s.Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>该模型 kind 下某参数的类型字符（'I'/'D'/'B'/'S'）。</summary>
        public char ParamType(string name)
        {
            var status = md_model_param_type((int)_kind, name, out var t);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException($"ParamType[{name}] failed: {GetLastError()}");
            return (char)t;
        }

        internal static string GetLastError()
        {
            var ptr = md_get_last_error();
            return ptr == IntPtr.Zero ? string.Empty : Utf8Helper.Read(ptr);
        }

        /// <summary>执行一次预测，返回结果句柄；调用方用完后必须 md_result_destroy。</summary>
        protected IntPtr PredictNative(IntPtr imageHandle)
        {
            var status = md_model_predict(_handle, imageHandle, out var result);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Predict failed: {GetLastError()}");
            return result;
        }

        /// <summary>预测并包装为 Prediction&lt;T&gt;：持句柄 + 惰性列表 + 句柄直达绘制。</summary>
        protected Prediction<T> MakePrediction<T>(VisionImage image, Func<IntPtr, T[]> reader)
        {
            return new Prediction<T>(PredictNative(image.Handle), reader);
        }

        /// <summary>批量预测：把多张图一次提交给原生 md_model_predict_batch，返回累积的结果句柄。</summary>
        protected IntPtr PredictBatchNative(IntPtr[] imgHandles)
        {
            var status = md_model_predict_batch(_handle, imgHandles, new UIntPtr((uint)imgHandles.Length), out var result);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"PredictBatch failed: {GetLastError()}");
            return result;
        }

        /// <summary>批量预测并包装为 Prediction&lt;T&gt;（平铺：所有图全部项合并；丢图片边界）。</summary>
        protected Prediction<T> PredictBatch<T>(IEnumerable<VisionImage> images, Func<IntPtr, T[]> reader)
        {
            if (images == null) throw new ArgumentNullException(nameof(images));
            var handles = new List<IntPtr>();
            foreach (var im in images)
            {
                if (im == null) throw new ArgumentNullException(nameof(images));
                handles.Add(im.Handle);
            }
            return new Prediction<T>(PredictBatchNative(handles.ToArray()), reader);
        }

        /// <summary>批量预测（2D）：每图一组托管数组（保留图片边界）。急切读取后立即释放原生结果句柄。</summary>
        protected IReadOnlyList<T[]> PredictBatch2D<T>(IEnumerable<VisionImage> images, Func<IntPtr, int, T[]> imageReader)
        {
            if (images == null) throw new ArgumentNullException(nameof(images));
            var handles = new List<IntPtr>();
            foreach (var im in images)
            {
                if (im == null) throw new ArgumentNullException(nameof(images));
                handles.Add(im.Handle);
            }
            if (handles.Count == 0) return Array.Empty<T[]>();
            IntPtr result = PredictBatchNative(handles.ToArray());
            try
            {
                var list = new List<T[]>(handles.Count);
                for (int i = 0; i < handles.Count; i++)
                    list.Add(imageReader(result, i));
                return list;
            }
            finally
            {
                md_result_destroy(result);
            }
        }

        /// <summary>深拷贝原生模型句柄（独立实例）。</summary>
        protected IntPtr CloneNative()
        {
            var status = md_model_clone(_handle, out var outHandle);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Clone failed: {GetLastError()}");
            return outHandle;
        }

        /// <summary>从原生句柄包装（子类 Clone 用）。</summary>
        protected BaseModel(MDModelKind kind, IntPtr existingHandle)
        {
            _kind = kind;
            _handle = existingHandle;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    md_model_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~BaseModel() => Dispose();
    }

    /// <summary>
    /// 内存归结果句柄所有的一次性读取器：封装"取数组指针 → 循环读结构 → 释放"。
    /// </summary>
    internal sealed class ResultReader : IDisposable
    {
        private readonly IntPtr _result;

        public ResultReader(IntPtr result) { _result = result; }
        public IntPtr Result => _result;

        /// <summary>数组 getter 委托（out 参数不允许出现在 Func 泛型，故自定义）。</summary>
        internal delegate MDStatus ItemGetter(IntPtr handle, out IntPtr items, out UIntPtr count);

        /// <summary>2D 批量数组 getter 委托：按图索引取每图项数组。</summary>
        internal delegate MDStatus BatchItemGetter(IntPtr handle, UIntPtr img, out IntPtr items, out UIntPtr count);

        public static T[] ReadItems<T>(IntPtr result, ItemGetter getter)
            where T : struct
        {
            var status = getter(result, out var items, out var count);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Result getter failed: {BaseModel.GetLastError()}");
            int n = checked((int)count);
            if (n == 0 || items == IntPtr.Zero) return Array.Empty<T>();
            var arr = new T[n];
            int stride = Marshal.SizeOf<T>();
            for (int i = 0; i < n; i++)
                arr[i] = Marshal.PtrToStructure<T>(IntPtr.Add(items, i * stride));
            return arr;
        }

        /// <summary>2D 批量：读第 img 张图的 blittable 项数组。</summary>
        public static T[] ReadItemsBatch<T>(IntPtr result, UIntPtr img, BatchItemGetter getter)
            where T : struct
        {
            var status = getter(result, img, out var items, out var count);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Batch getter failed: {BaseModel.GetLastError()}");
            int n = checked((int)count);
            if (n == 0 || items == IntPtr.Zero) return Array.Empty<T>();
            var arr = new T[n];
            int stride = Marshal.SizeOf<T>();
            for (int i = 0; i < n; i++)
                arr[i] = Marshal.PtrToStructure<T>(IntPtr.Add(items, i * stride));
            return arr;
        }

        public static byte[] ReadBytes(IntPtr ptr, UIntPtr count)
        {
            int n = checked((int)count);
            if (n == 0 || ptr == IntPtr.Zero) return Array.Empty<byte>();
            var buf = new byte[n];
            Marshal.Copy(ptr, buf, 0, n);
            return buf;
        }

        public static float[] ReadFloats(IntPtr ptr, UIntPtr count)
        {
            int n = checked((int)count);
            if (n == 0 || ptr == IntPtr.Zero) return Array.Empty<float>();
            var buf = new float[n];
            Marshal.Copy(ptr, buf, 0, n);
            return buf;
        }

        public static string ReadString(IntPtr ptr)
        {
            return ptr == IntPtr.Zero ? null : Utf8Helper.Read(ptr);
        }

        public void Dispose()
        {
            if (_result != IntPtr.Zero)
            {
                md_result_destroy(_result);
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>UTF-8 字符串读取/分配辅助（netstandard2.0 无 Marshal.PtrToStringUTF8）。</summary>
    internal static class Utf8Helper
    {
        public static string Read(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return string.Empty;
            int len = 0;
            while (Marshal.ReadByte(ptr, len) != 0) len++;
            if (len == 0) return string.Empty;
            var bytes = new byte[len];
            Marshal.Copy(ptr, bytes, 0, len);
            return System.Text.Encoding.UTF8.GetString(bytes);
        }

        /// <summary>分配 NUL 结尾的 UTF-8 内存；调用方负责 Marshal.FreeHGlobal。</summary>
        public static IntPtr Alloc(string s)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
            var ptr = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, ptr, bytes.Length);
            return ptr;
        }
    }
}
