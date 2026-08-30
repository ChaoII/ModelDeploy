using ModelDeploy.Solutions;

namespace ModelDeployUnitTest;

/// <summary>
/// 解决方案层（vision::solution / tool）C# 封装测试：
/// ObjectCounter 跨线计数 + Heatmap 峰点 + Iou。
/// </summary>
public class CapisolutionTests
{
    [Test]
    public void ObjectCounter_CrossesLine_CountsIn()
    {
        using var counter = new ObjectCounter();
        counter.SetLine(5f, 0f, 5f, 10f);

        // 框中心 x=1（线左）
        counter.Update(new float[] { 0, 4, 2, 2 }, new[] { 0 }, new[] { 1 });
        Assert.That(counter.HLine().In, Is.EqualTo(0));

        // 框中心 x=9（线右）
        counter.Update(new float[] { 8, 4, 2, 2 }, new[] { 0 }, new[] { 1 });
        Assert.That(counter.HLine().In, Is.EqualTo(1));
    }

    [Test]
    public void Heatmap_SetSizeUpdate_ReturnsPeak()
    {
        using var heat = new Heatmap();
        heat.SetSize(10, 10);
        heat.Update(new float[] { 2, 2, 2, 2, 7, 7, 2, 2 }, 10, 10);
        var (x, y) = heat.Peak();
        Assert.That(x, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void Tool_Iou_Identical_IsOne()
    {
        Assert.That(Tool.Iou(0, 0, 10, 10, 0, 0, 10, 10), Is.EqualTo(1.0f).Within(1e-5f));
    }

    [Test]
    public void RegionCounter_CountsInsidePolygon()
    {
        var poly = new float[] { 0, 0, 10, 0, 10, 10, 0, 10 };
        // 框中心 (3,3) 在区域内；(51,51) 在区域外
        var boxes = new float[] { 2, 2, 2, 2, 50, 50, 2, 2 };
        var ids = new[] { 1, 2 };
        var labels = new[] { 0, 0 };

        using var rc = new RegionCounter();
        rc.AddRegion("A", poly);
        rc.Update(boxes, ids, labels);
        Assert.That(rc.Count("A"), Is.EqualTo(1));
    }

    [Test]
    public void QueueManager_CountsInsideRegion()
    {
        var poly = new float[] { 0, 0, 10, 0, 10, 10, 0, 10 };
        var boxes = new float[] { 2, 2, 2, 2, 50, 50, 2, 2 };
        var ids = new[] { 1, 2 };
        var labels = new[] { 0, 0 };

        using var q = new QueueManager();
        q.SetRegion(poly);
        q.Update(boxes, ids, labels);
        Assert.That(q.Count(), Is.EqualTo(1));
    }

    [Test]
    public void TrackZone_CountsInsideRegion()
    {
        var poly = new float[] { 0, 0, 10, 0, 10, 10, 0, 10 };
        var boxes = new float[] { 2, 2, 2, 2, 50, 50, 2, 2 };
        var ids = new[] { 1, 2 };
        var labels = new[] { 0, 0 };

        using var tz = new TrackZone();
        tz.SetRegion(poly);
        tz.Update(boxes, ids, labels);
        Assert.That(tz.Count(), Is.EqualTo(1));
    }
}
