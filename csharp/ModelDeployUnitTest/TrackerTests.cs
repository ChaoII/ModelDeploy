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
}
