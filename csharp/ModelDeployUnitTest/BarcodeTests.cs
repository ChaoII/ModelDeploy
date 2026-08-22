using ModelDeploy;

namespace ModelDeployUnitTest;

/// <summary>
/// 条码 / 二维码识别器（纯 CV，无模型依赖）。
/// </summary>
public class BarcodeTests
{
    private static string GetTestDataDir()
    {
        return Environment.GetEnvironmentVariable("TEST_DATA_DIR")
            ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
    }

    [Test]
    public void BarcodeDetect_QR_ReturnsText()
    {
        var path = Path.Combine(GetTestDataDir(), "test_data", "qr_sample.png");
        Assert.That(File.Exists(path), Is.True, $"qr_sample.png not found at {path}");

        using var det = new BarcodeDetector();
        using var img = VisionImage.Read(path);

        var res = det.Detect(img);

        Assert.That(res, Is.Not.Empty);
        Assert.Multiple(() =>
        {
            Assert.That(res[0].IsQr, Is.True);
            Assert.That(res[0].Text, Is.EqualTo("https://example.com/MD"));
        });
    }

    [Test]
    public void BarcodeDetect_NonBarcodeImage_ReturnsEmpty()
    {
        // 用纯色图像反证：无条码时返回空数组而不抛异常。
        var raw = new byte[100 * 100 * 3];
        using var img = VisionImage.FromRgb24Data(raw, 100, 100);
        using var det = new BarcodeDetector();

        var res = det.Detect(img);

        Assert.That(res, Is.Empty);
    }

    [Test]
    public void SetFormats_NoThrow()
    {
        using var det = new BarcodeDetector();
        Assert.DoesNotThrow(() => det.SetFormats(0));
    }
}
