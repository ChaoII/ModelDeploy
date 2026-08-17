using ModelDeploy;
using ModelDeploy.V2;
using ModelDeploy.V2.Models;
using ModelDeploy.V2.Results;
using NUnit.Framework;

namespace ModelDeployUnitTest;

public class Capi2VisionTests
{
    private static string GetTestDataDir()
    {
        return Environment.GetEnvironmentVariable("TEST_DATA_DIR")
            ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
    }

    private static string ModelRoot => Path.Combine(GetTestDataDir(), "test_data", "test_models", "onnx");
    private static string ImageRoot => Path.Combine(GetTestDataDir(), "test_data", "test_images");

    private static bool Has(string p) => File.Exists(p);

    private static RuntimeOption2 CpuOrt()
    {
        return new RuntimeOption2().UseOrt().SetDevice(Device.CPU).SetCpuThreads(4);
    }

    // ==================== VisionImage ====================

    [Test]
    public void VisionImage_Read_SetsSize()
    {
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(img)) Assert.Ignore("image not found");
        using var vi = VisionImage.Read(img);
        Assert.That(vi.Width, Is.GreaterThan(0));
        Assert.That(vi.Height, Is.GreaterThan(0));
        Assert.That(vi.Channels, Is.EqualTo(3));
    }

    [Test]
    public void VisionImage_Clone_Crop_Save()
    {
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(img)) Assert.Ignore("image not found");
        using var vi = VisionImage.Read(img);
        using var clone = vi.Clone();
        Assert.That(clone.Width, Is.EqualTo(vi.Width));
        using var crop = vi.Crop(10, 10, 50, 50);
        Assert.That(crop.Width, Is.EqualTo(50));
        Assert.That(crop.Height, Is.EqualTo(50));
        var tmp = Path.Combine(Path.GetTempPath(), "md_v2_save.png");
        try
        {
            Assert.DoesNotThrow(() => vi.Save(tmp));
            Assert.That(new FileInfo(tmp).Length, Is.GreaterThan(0));
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Test]
    public void VisionImage_Encode_ReturnsBytes()
    {
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(img)) Assert.Ignore("image not found");
        using var vi = VisionImage.Read(img);
        var jpg = vi.Encode(".jpg");
        Assert.That(jpg.Length, Is.GreaterThan(0));
        // jpg 以 FF D8 开头
        Assert.That(jpg[0], Is.EqualTo(0xFF));
        Assert.That(jpg[1], Is.EqualTo(0xD8));
    }

    [Test]
    public void VisionImage_FromBgr24_Works()
    {
        using var vi = VisionImage.FromBgr24Data(new byte[100 * 80 * 3], 100, 80);
        Assert.That(vi.Width, Is.EqualTo(100));
        Assert.That(vi.Height, Is.EqualTo(80));
    }

    // ==================== Detection ====================

    [Test]
    public void Detection_Predict_ReturnsObjects()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("model/image not found");

        using var vi = VisionImage.Read(img);
        using var det = new DetectionModel(model, CpuOrt());
        var results = det.Predict(vi);

        Assert.That(results, Is.Not.Empty);
        foreach (var r in results)
        {
            Assert.That(r.Score, Is.GreaterThan(0f).And.LessThanOrEqualTo(1f));
            Assert.That(r.Box.Width, Is.GreaterThan(0));
        }
    }

    // ==================== InsightFace ====================

    [Test]
    public void InsightFace_Predict_ReturnsFaces()
    {
        var dir = Path.Combine(ModelRoot, "insightface", "buffalo_l");
        var model = string.Join("|",
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            Path.Combine(dir, "genderage.onnx"));
        var img = Path.Combine(ImageRoot, "test_face1.jpg");
        if (!Has(Path.Combine(dir, "det_10g.onnx")) || !Has(img)) Assert.Ignore("model/image not found");

        using var vi = VisionImage.Read(img);
        using var modelObj = new InsightFaceModel(model, CpuOrt());
        var results = modelObj.Predict(vi);

        Assert.That(results, Is.Not.Empty);
        var first = results[0];
        Assert.That(first.Embedding.Length, Is.GreaterThan(0), "should have embedding");
        Assert.That(first.Gender, Is.GreaterThanOrEqualTo(0));
    }

    // ==================== OCR ====================

    [Test]
    public void Ocr_Predict_ReturnsText()
    {
        var dir = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile");
        var dict = Path.Combine(GetTestDataDir(), "test_data", "ppocrv4_dict.txt");
        var model = string.Join("|",
            Path.Combine(dir, "det_infer.onnx"),
            Path.Combine(dir, "cls_infer.onnx"),
            Path.Combine(dir, "rec_infer.onnx"),
            dict);
        var img = Path.Combine(ImageRoot, "test_ocr.png");
        if (!Has(Path.Combine(dir, "det_infer.onnx")) || !Has(img)) Assert.Ignore("model/image not found");

        using var vi = VisionImage.Read(img);
        using var ocr = new OcrModel(model, CpuOrt());
        var results = ocr.Predict(vi);

        Assert.That(results, Is.Not.Empty);
        Assert.That(results[0].Text, Is.Not.Empty);
    }

    // ==================== Draw ====================

    [Test]
    public void Draw_Rect_ModifiesPixels()
    {
        using var vi = VisionImage.FromBgr24Data(new byte[50 * 50 * 3], 50, 50);
        Assert.DoesNotThrow(() => Draw.DrawRect(vi, 5, 5, 20, 20, 255, 0, 0, 1.0f));
        Assert.DoesNotThrow(() => Draw.DrawText(vi, 5, 30, "hi", null, 20, 255, 255, 255));
    }
}
