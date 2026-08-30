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

    /// <summary>命名区域计数（对应 CAPI md_solution_region_counter_*）。</summary>
    public sealed class RegionCounter : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public RegionCounter()
        {
            if (md_solution_create(out _handle, MDSolutionKind.MD_SOLUTION_REGION_COUNTER) != MDStatus.MD_OK)
                throw new SolutionException("md_solution_create(RegionCounter) failed");
        }

        /// <summary>polygon 为扁平 [x0,y0,x1,y1,...]。</summary>
        public void AddRegion(string name, float[] xy)
        {
            if (name == null) throw new ArgumentNullException(nameof(name));
            if (xy == null || xy.Length < 2 || xy.Length % 2 != 0)
                throw new ArgumentException("xy must contain an even number of values (x,y pairs)");
            md_solution_region_counter_add(_handle, name, xy, new UIntPtr((uint)(xy.Length / 2)));
        }

        /// <summary>boxes 为扁平 [x,y,w,h,...]；ids/labels 与框数等长（C 顺序为 track_ids, label_ids）。</summary>
        public void Update(float[] boxes, int[] ids, int[] labels)
        {
            if (boxes == null || ids == null || labels == null)
                throw new ArgumentNullException(nameof(boxes));
            if (ids.Length == 0 || boxes.Length < ids.Length * 4)
                throw new ArgumentException("boxes must contain n*4 values for n boxes");
            md_solution_region_counter_update(_handle, boxes,
                new UIntPtr((uint)(boxes.Length / 4)), ids, labels);
        }

        public int Count(string name) =>
            name == null ? 0 : md_solution_region_counter_count(_handle, name);

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

    /// <summary>排队区域计数（对应 CAPI md_solution_queue_*）。</summary>
    public sealed class QueueManager : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public QueueManager()
        {
            if (md_solution_create(out _handle, MDSolutionKind.MD_SOLUTION_QUEUE) != MDStatus.MD_OK)
                throw new SolutionException("md_solution_create(QueueManager) failed");
        }

        /// <summary>polygon 为扁平 [x0,y0,x1,y1,...]。</summary>
        public void SetRegion(float[] xy)
        {
            if (xy == null || xy.Length < 2 || xy.Length % 2 != 0)
                throw new ArgumentException("xy must contain an even number of values (x,y pairs)");
            md_solution_queue_set_region(_handle, xy, new UIntPtr((uint)(xy.Length / 2)));
        }

        /// <summary>boxes 为扁平 [x,y,w,h,...]；ids/labels 与框数等长（C 顺序为 track_ids, label_ids）。</summary>
        public void Update(float[] boxes, int[] ids, int[] labels)
        {
            if (boxes == null || ids == null || labels == null)
                throw new ArgumentNullException(nameof(boxes));
            if (ids.Length == 0 || boxes.Length < ids.Length * 4)
                throw new ArgumentException("boxes must contain n*4 values for n boxes");
            md_solution_queue_update(_handle, boxes,
                new UIntPtr((uint)(boxes.Length / 4)), ids, labels);
        }

        public int Count() => md_solution_queue_count(_handle);

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

    /// <summary>区域轨迹保留并计数（对应 CAPI md_solution_track_zone_*）。</summary>
    public sealed class TrackZone : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public TrackZone()
        {
            if (md_solution_create(out _handle, MDSolutionKind.MD_SOLUTION_TRACK_ZONE) != MDStatus.MD_OK)
                throw new SolutionException("md_solution_create(TrackZone) failed");
        }

        /// <summary>polygon 为扁平 [x0,y0,x1,y1,...]。</summary>
        public void SetRegion(float[] xy)
        {
            if (xy == null || xy.Length < 2 || xy.Length % 2 != 0)
                throw new ArgumentException("xy must contain an even number of values (x,y pairs)");
            md_solution_track_zone_set_region(_handle, xy, new UIntPtr((uint)(xy.Length / 2)));
        }

        /// <summary>boxes 为扁平 [x,y,w,h,...]；ids/labels 与框数等长（C 顺序为 track_ids, label_ids）。</summary>
        public void Update(float[] boxes, int[] ids, int[] labels)
        {
            if (boxes == null || ids == null || labels == null)
                throw new ArgumentNullException(nameof(boxes));
            if (ids.Length == 0 || boxes.Length < ids.Length * 4)
                throw new ArgumentException("boxes must contain n*4 values for n boxes");
            md_solution_track_zone_update(_handle, boxes,
                new UIntPtr((uint)(boxes.Length / 4)), ids, labels);
        }

        public int Count() => md_solution_track_zone_count(_handle);

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
