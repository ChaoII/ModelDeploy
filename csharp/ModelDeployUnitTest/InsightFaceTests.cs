using ModelDeploy;
using ModelDeploy.utils;
using ModelDeploy.vision.face;

namespace ModelDeployUnitTest;

public class InsightFaceTests
{
    private static string TestDataDir =>
        Environment.GetEnvironmentVariable("TEST_DATA_DIR")
        ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");

    private static string TestImagePath => Path.Combine(TestDataDir, "test_data", "test_images", "test_person.jpg");

    private static bool HasTestImage => File.Exists(TestImagePath);

    private static bool HasModels
    {
        get
        {
            var dir = Path.Combine(TestDataDir, "test_data", "test_models", "onnx", "insightface", "buffalo_l");
            return File.Exists(Path.Combine(dir, "det_10g.onnx"))
                && File.Exists(Path.Combine(dir, "w600k_r50.onnx"))
                && File.Exists(Path.Combine(dir, "2d106det.onnx"))
                && File.Exists(Path.Combine(dir, "1k3d68.onnx"));
        }
    }

    [Test]
    public void InsightFace_Analyze_ReturnsFaces()
    {
        if (!HasModels) Assert.Ignore("insightface models not found");
        if (!HasTestImage) Assert.Ignore("test image not found");

        var dir = Path.Combine(TestDataDir, "test_data", "test_models", "onnx", "insightface", "buffalo_l");
        using var img = Image.Read(TestImagePath);
        var opt = new RuntimeOption();
        using var model = new InsightFaceAnalysis(
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            opt);

        var results = model.Analyze(img);
        Assert.That(results, Is.Not.Empty, "insightface should detect faces");
        foreach (var r in results)
        {
            Assert.That(r.Score, Is.GreaterThan(0f).And.LessThanOrEqualTo(1f));
            if (r.Embedding.Length > 0)
            {
                Assert.That(r.Embedding.Length, Is.EqualTo(512));
            }
        }
    }

    [Test]
    public void InsightFace_SetDetThresh_DoesNotThrow()
    {
        if (!HasModels) Assert.Ignore("insightface models not found");
        var dir = Path.Combine(TestDataDir, "test_data", "test_models", "onnx", "insightface", "buffalo_l");
        var opt = new RuntimeOption();
        using var model = new InsightFaceAnalysis(
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            opt);
        Assert.DoesNotThrow(() => model.SetDetThresh(0.5f));
    }

    [Test]
    public void InsightFace_NonExistentModels_Throws()
    {
        var opt = new RuntimeOption();
        Assert.Throws<InvalidOperationException>(() => _ = new InsightFaceAnalysis(
            "/nonexistent/det_10g.onnx", "/nonexistent/w600k_r50.onnx",
            "/nonexistent/2d106det.onnx", "/nonexistent/1k3d68.onnx", opt));
    }
}
