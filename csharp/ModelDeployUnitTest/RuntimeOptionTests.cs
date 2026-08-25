using ModelDeploy;

namespace ModelDeployUnitTest;

/// <summary>
/// capi RuntimeOption 链式 setter 接线测试（不依赖模型/数据）。
/// </summary>
public class RuntimeOptionTests
{
    [Test]
    public void SetDevice_Gpu_Id_Applies()
    {
        using var opt = new RuntimeOption();
        opt.SetDevice(Device.GPU, 1).UseOrt();
        Assert.Pass("composition ok (no native error)");
    }

    [Test]
    public void SetConfig_TrI_CachePath_Ok()
    {
        using var opt = new RuntimeOption();
        opt.SetConfig("ort", "trt_engine_cache_path", "./x");
        Assert.Pass("config accepted");
    }

    [Test]
    public void SetConfig_UnknownKey_Throws()
    {
        using var opt = new RuntimeOption();
        Assert.Throws<InvalidOperationException>(() => opt.SetConfig("bogus", "k", "v"));
    }
}
