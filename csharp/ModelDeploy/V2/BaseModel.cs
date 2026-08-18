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

        /// <summary>
        /// NV12 直接输入推理（硬解码/摄像头直通，srcDevice 指明 Y/UV 内存所在设备；
        /// 经 md_image_from_device_nv12 包装为设备帧 ImageData 后走统一 predict(ImageData) 单入口）。
        /// </summary>
        protected Prediction<T> MakePredictionNv12<T>(byte[] y, byte[] uv,
            int w, int h, int stepY, int stepUv, Device srcDevice, Func<IntPtr, T[]> reader)
        {
            var (pred, frame) = MakePredictionNv12WithFrame<T>(y, uv, w, h, stepY, stepUv, srcDevice, reader);
            frame?.Dispose();
            return pred;
        }

        /// <summary>
        /// NV12 直接输入推理并返回绑定的输入帧（设备相关的 ImageData 包装）。
        /// 每步：md_image_from_device_nv12 构造设备帧 → md_model_predict（统一单入口）。
        /// 可从 Frame 取 Y/UV 平面指针或就地绘制；Frame 由调用方负责 Dispose，
        /// Prediction 也需 Dispose 释放结果句柄。
        /// </summary>
        protected (Prediction<T> Prediction, VisionImage Frame) MakePredictionNv12WithFrame<T>(
            byte[] y, byte[] uv,
            int w, int h, int stepY, int stepUv, Device srcDevice, Func<IntPtr, T[]> reader)
        {
            var status = md_image_from_device_nv12(out var img, y, uv, w, h,
                stepY, stepUv, (int)srcDevice);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"MakePredictionNv12: md_image_from_device_nv12 failed: {GetLastError()}");
            var frameImg = VisionImage.FromDeviceFrame(img);
            frameImg.PinBuffers(y, uv);
            Prediction<T> prediction;
            try
            {
                prediction = new Prediction<T>(PredictNative(img), reader);
            }
            catch
            {
                frameImg.Dispose();
                throw;
            }
            return (prediction, frameImg);
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
