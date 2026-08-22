using ModelDeploy;
using ModelDeploy.Tracking;

namespace ModelDeployUnitTest;

/// <summary>
/// 多目标跟踪器（CPU，无模型/数据依赖）。
/// </summary>
public class TrackerTests
{
    [Test]
    public void ByteTrack_StableTrackId_OverFrames()
    {
        using var tracker = new Tracker(TrackerKind.ByteTrack);

        var frame1 = tracker.Update(
            new[] { new RectF(10, 20, 30, 40) },
            new[] { 0.9f },
            new[] { 0 });
        Assert.That(frame1, Is.Not.Empty);

        var id1 = frame1[0].TrackId;

        var frame2 = tracker.Update(
            new[] { new RectF(14, 24, 30, 40) },
            new[] { 0.91f },
            new[] { 0 });
        Assert.That(frame2, Is.Not.Empty);

        Assert.That(frame2[0].TrackId, Is.EqualTo(id1));
    }

    [Test]
    public void ByteTrack_EmptyInput_ReturnsEmpty()
    {
        using var tracker = new Tracker(TrackerKind.ByteTrack);
        var frame = tracker.Update(
            Array.Empty<RectF>(),
            Array.Empty<float>(),
            Array.Empty<int>());
        Assert.That(frame, Is.Empty);
    }

    [Test]
    public void Reset_ClearsState()
    {
        using var tracker = new Tracker(TrackerKind.ByteTrack);
        tracker.Update(
            new[] { new RectF(10, 20, 30, 40) },
            new[] { 0.9f },
            new[] { 0 });
        tracker.Reset();
        Assert.Pass("Reset completed without error");
    }

    // F1 回归：Update 遵循 查询(capacity,非变异)→分配→单次提交 契约，每逻辑帧只推进一次。
    // 若仍用 Update 做两阶段探测（双重推进），Lost 目标的 max_age 每帧 +2、有效减半并被过早
    // 移除；本测试让 A 在 max_age 内于"另一目标持续出现"环境下重新出现，断言其 ID 保持不变。
    [Test]
    public void ByteTrack_NoDoubleAdvance_KeepsLostTrackId()
    {
        using var tracker = new Tracker(TrackerKind.ByteTrack);
        tracker.SetParam("max_age", 30);

        var a = new RectF(100, 100, 40, 40);
        var a2 = new RectF(102, 102, 40, 40);
        var frame1 = tracker.Update(new[] { a }, new[] { 0.95f }, new[] { 0 });
        Assert.That(frame1, Is.Not.Empty);
        int aId = frame1[0].TrackId;
        tracker.Update(new[] { a2 }, new[] { 0.95f }, new[] { 0 });

        var b = new RectF(400, 400, 40, 40);
        for (int i = 0; i < 23; i++)
        {
            var rb = tracker.Update(new[] { b }, new[] { 0.95f }, new[] { 1 });
            Assert.That(rb, Is.Not.Empty, "B 每帧都应被跟踪");
        }

        var reappear = tracker.Update(new[] { a, b }, new[] { 0.95f, 0.95f }, new[] { 0, 1 });
        int aIdx = -1;
        for (int i = 0; i < reappear.Length; i++)
            if (reappear[i].X == 100 && reappear[i].Y == 100) { aIdx = i; break; }
        Assert.That(aIdx, Is.GreaterThanOrEqualTo(0), "A 重新出现时应被跟踪");
        Assert.That(reappear[aIdx].TrackId, Is.EqualTo(aId), "A 的 ID 应保持不变（无双重推进）");
    }
}
