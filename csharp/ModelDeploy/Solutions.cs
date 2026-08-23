using System;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.Solutions
{
    /// <summary>解决方案创建失败抛出。</summary>
    public sealed class SolutionException : Exception
    {
        public SolutionException(string message) : base(message) { }
    }

    /// <summary>跨线目标计数（对应 CAPI md_solution_object_counter_*）。</summary>
    public sealed class ObjectCounter : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public ObjectCounter()
        {
            if (md_solution_create(out _handle, MDSolutionKind.MD_SOLUTION_OBJECT_COUNTER) != MDStatus.MD_OK)
                throw new SolutionException("md_solution_create(ObjectCounter) failed");
        }

        public void SetLine(float ax, float ay, float bx, float by) =>
            md_solution_object_counter_set_line(_handle, ax, ay, bx, by);

        public void Update(float[] boxes, int[] labelIds, int[] trackIds)
        {
            if (boxes == null || labelIds == null || trackIds == null)
                throw new ArgumentNullException(nameof(boxes));
            if (labelIds.Length == 0 || boxes.Length < labelIds.Length * 4)
                throw new ArgumentException("boxes must contain n*4 values for n labels");
            md_solution_object_counter_update(_handle, boxes,
                new UIntPtr((uint)labelIds.Length), labelIds, trackIds);
        }

        public (int In, int Out) HLine()
        {
            md_solution_object_counter_hline(_handle, out int i, out int o);
            return (i, o);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero && !_disposed)
            {
                md_solution_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }
    }

    /// <summary>轨迹热力图（对应 CAPI md_solution_heatmap_*）。</summary>
    public sealed class Heatmap : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public Heatmap()
        {
            if (md_solution_create(out _handle, MDSolutionKind.MD_SOLUTION_HEATMAP) != MDStatus.MD_OK)
                throw new SolutionException("md_solution_create(Heatmap) failed");
        }

        public void SetSize(int w, int h) => md_solution_heatmap_set_size(_handle, w, h);

        public void Update(float[] boxes, int frameW, int frameH)
        {
            if (boxes == null) throw new ArgumentNullException(nameof(boxes));
            md_solution_heatmap_update(_handle, boxes, new UIntPtr((uint)(boxes.Length / 4)), frameW, frameH);
        }

        public (int X, int Y) Peak()
        {
            md_solution_heatmap_peak(_handle, out int x, out int y);
            return (x, y);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero && !_disposed)
            {
                md_solution_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }
    }

    /// <summary>纯工具函数。</summary>
    public static class Tool
    {
        /// <summary>两个矩形（x,y,w,h）的交并比。</summary>
        public static float Iou(float ax, float ay, float aw, float ah,
                                float bx, float by, float bw, float bh)
        {
            md_vision_iou4(ax, ay, aw, ah, bx, by, bw, bh, out float iou);
            return iou;
        }
    }
}
