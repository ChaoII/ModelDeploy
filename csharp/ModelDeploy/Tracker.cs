using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.Tracking
{
    /// <summary>跟踪器类型（对应 capi MDTrackerKind）。</summary>
    public enum TrackerKind
    {
        ByteTrack = 0,
        BotSort = 1,
        StrongSort = 2
    }

    /// <summary>跟踪目标（对应 capi MDTrackItem）。状态：New=0/Tracked=1/Lost=2/Removed=3。</summary>
    public readonly struct TrackItem
    {
        public float X { get; }
        public float Y { get; }
        public float Width { get; }
        public float Height { get; }
        public int TrackId { get; }
        public int LabelId { get; }
        public float Score { get; }
        public int State { get; }

        public TrackItem(float x, float y, float w, float h, int trackId, int labelId, float score, int state)
        {
            X = x; Y = y; Width = w; Height = h;
            TrackId = trackId; LabelId = labelId; Score = score; State = state;
        }

        public override string ToString() => $"Track#{TrackId}({X},{Y},{Width}x{Height} score={Score} state={State})";
    }

    /// <summary>
    /// 多目标跟踪器（ByteTrack / BoT-SORT / StrongSORT）。
    /// 纯 CPU 无模型依赖；跟踪 ID 跨帧稳定，Reset 归零。
    /// </summary>
    public sealed class Tracker : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>按类型创建跟踪器。</summary>
        public Tracker(TrackerKind kind)
        {
            var status = md_tracker_create((MDTrackerKind)(int)kind, out _handle);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Tracker create failed: {BaseModel.GetLastError()}");
        }

        /// <summary>逐帧更新：输入检测框/得分/类别，返回本帧跟踪目标（TrackId 跨帧稳定）。</summary>
        /// <remarks>
        /// 遵循 capi 查询/提交契约：先以 md_tracker_capacity 做非变异容量查询（不推进状态），
        /// 分配所需缓冲后再以恰好容量调用有状态 md_tracker_update 一次。这样每逻辑帧跟踪器只
        /// 推进一次（Kalman / max_age / ID 计数正确），避免旧的双阶段探测导致的双重推进。
        /// </remarks>
        public TrackItem[] Update(RectF[] boxes, float[] scores, int[] labelIds)
        {
            ThrowIfDisposed();
            if (boxes == null) throw new ArgumentNullException(nameof(boxes));
            int n = boxes.Length;
            if (scores == null || scores.Length != n)
                throw new ArgumentException("scores length must match boxes", nameof(scores));
            if (labelIds == null || labelIds.Length != n)
                throw new ArgumentException("labelIds length must match boxes", nameof(labelIds));

            var mdBoxes = new MDBox[n];
            for (int i = 0; i < n; i++)
            {
                mdBoxes[i] = new MDBox
                {
                    x = boxes[i].X,
                    y = boxes[i].Y,
                    w = boxes[i].Width,
                    h = boxes[i].Height
                };
            }

            var nU = new UIntPtr((uint)n);

            // 阶段 1：非变异容量查询（不推进跟踪器状态）
            if (md_tracker_capacity(_handle, mdBoxes, scores, labelIds, nU, out UIntPtr need) != MDStatus.MD_OK)
                throw new InvalidOperationException($"Tracker capacity query failed: {BaseModel.GetLastError()}");

            // 阶段 2：分配 needed 个目标，正式写入（有状态，恰一次）。
            // 注意：即使 need==0（空帧）也要调用一次 update 以推进状态；此时仍给一个
            // 最小为 1 的非空缓冲，避免空数组被 marshal 成 null 触发 native 空指针守卫。
            uint cap = Math.Max(checked((uint)need), 1);
            var buf = new MDTrackItem[cap];
            var outCount = new UIntPtr(cap);
            var status = md_tracker_update(_handle, mdBoxes, scores, labelIds, nU, buf, ref outCount);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Tracker update failed: {BaseModel.GetLastError()}");
            int written = checked((int)outCount);

            var result = new TrackItem[written];
            for (int i = 0; i < written; i++)
                result[i] = ToTrackItem(buf[i]);
            return result;
        }

        /// <summary>设置跟踪参数（track_thresh / max_age / iou_threshold ...，双精度命名参数）。</summary>
        public void SetParam(string name, double value)
        {
            ThrowIfDisposed();
            var status = md_tracker_set_params(_handle, name, value);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"SetParam[{name}] failed: {BaseModel.GetLastError()}");
        }

        /// <summary>重置内部跟踪状态（ID 计数归零）。</summary>
        public void Reset()
        {
            ThrowIfDisposed();
            var status = md_tracker_reset(_handle);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Tracker reset failed: {BaseModel.GetLastError()}");
        }

        private static TrackItem ToTrackItem(MDTrackItem it)
        {
            return new TrackItem(it.x, it.y, it.w, it.h, it.track_id, it.label_id, it.score, it.state);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Tracker));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    md_tracker_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }

        ~Tracker() => Dispose();
    }
}
