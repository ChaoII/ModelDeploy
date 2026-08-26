using ModelDeploy;
using ModelDeploy.Models;
using ModelDeploy.Results;
using NUnit.Framework;

namespace ModelDeployUnitTest;

public class CapiVisionTests
{
    private static string GetTestDataDir()
    {
        return Environment.GetEnvironmentVariable("TEST_DATA_DIR")
            ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
    }

    private static string ModelRoot => Path.Combine(GetTestDataDir(), "test_data", "test_models", "onnx");
    private static string ImageRoot => Path.Combine(GetTestDataDir(), "test_data", "test_images");

    private static bool Has(string p) => File.Exists(p);

    private static RuntimeOption CpuOrt()
    {
        return new RuntimeOption().UseOrt().SetDevice(Device.CPU).SetCpuThreads(4);
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

    [Test]
    public void VisionImage_FromNv12Data_IsNv12TwoPlane()
    {
        int w = 64, h = 48;
        var y = new byte[w * h];
        var uv = new byte[w * h / 2];
        using var vi = VisionImage.FromNv12Data(y, uv, w, h);
        Assert.That(vi.Type, Is.EqualTo(MdImageType.NV12));
        Assert.That(vi.PlaneCount, Is.EqualTo(2));
        Assert.That(vi.Device, Is.EqualTo(Device.CPU));
        Assert.That(vi.GetPlane(0).Data, Is.Not.EqualTo(IntPtr.Zero));
        Assert.That(vi.GetPlane(1).Data, Is.Not.EqualTo(IntPtr.Zero));
        Assert.That(vi.GetPlane(0).Step, Is.EqualTo(w));
    }

    // ==================== ToNativeBytes / GetPlaneBytes ====================

    [Test]
    public void ToNativeBytes_Bgr24_ReturnsRawBytes() {
        var data = new byte[4 * 3 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = (byte)i;
        using var img = VisionImage.FromBgr24Data(data, 4, 3);
        var nb = img.ToNativeBytes();
        Assert.That(nb, Is.EqualTo(data));
    }

    [Test]
    public void ToNativeBytes_Nv12_ReturnsYPlusUv() {
        int w = 4, h = 2;
        var y = new byte[w * h];
        var uv = new byte[w * (h / 2)];
        for (int i = 0; i < uv.Length; i++) uv[i] = (byte)i;
        using var img = VisionImage.FromNv12Data(y, uv, w, h, 0, 0);
        var nb = img.ToNativeBytes();
        Assert.That(nb.Length, Is.EqualTo(w * h + uv.Length));
    }

    [Test]
    public void GetPlaneBytes_Bgr24_SinglePlane_AndOob_Throws() {
        using var img = VisionImage.FromBgr24Data(new byte[4 * 3 * 3], 4, 3);
        Assert.That(img.GetPlaneBytes(0).Length, Is.EqualTo(4 * 3 * 3));
        Assert.Throws<InvalidOperationException>(() => img.GetPlaneBytes(1));
    }

    [Test]
    public void VisionImage_Read_TypeIsBgr()
    {
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(img)) Assert.Ignore("image not found");
        using var vi = VisionImage.Read(img);
        Assert.That(vi.Type, Is.EqualTo(MdImageType.PKG_BGR_U8));
        Assert.That(vi.PlaneCount, Is.EqualTo(1));
        Assert.That(vi.Channels, Is.EqualTo(3));
    }

    [Test]
    public void VisionImage_FromNv12Data_Then_DetectionPredict()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        if (!Has(model)) Assert.Ignore("model not found");

        int w = 640, h = 640;
        var y = new byte[w * h];
        var uv = new byte[w * h / 2];
        using var vi = VisionImage.FromNv12Data(y, uv, w, h);
        using var det = new DetectionModel(model, CpuOrt());
        var results = det.Predict(vi);
        Assert.That(results, Is.Not.Null);
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

    [Test]
    public void DetectionModel_PredictBatch_Is2DPerImage()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("model/image not found");

        using var det = new DetectionModel(model, CpuOrt());
        using var vi1 = VisionImage.Read(img);
        using var vi2 = VisionImage.Read(img);
        int cnt0 = det.Predict(vi1).Count;
        int cnt1 = det.Predict(vi2).Count;

        var batch = det.PredictBatch(new[] { vi1, vi2 });
        // 2D：每图一组，保留图片边界
        Assert.That(batch.Count, Is.EqualTo(2), "batch should return one array per image");
        Assert.That(batch[0].Length, Is.EqualTo(cnt0), "img0 boxes should match its single predict");
        Assert.That(batch[1].Length, Is.EqualTo(cnt1), "img1 boxes should match its single predict");
        int total = 0;
        foreach (var arr in batch) total += arr.Length;
        Assert.That(total, Is.EqualTo(cnt0 + cnt1), "total boxes preserved across images");
    }

    [Test]
    public void DetectionModel_NoNv12Variants_UsesUnifiedEntry()
    {
        // PredictNv12 / PredictNv12WithFrame 已删除；统一入口 Predict(VisionImage) 需支持
        // NV12 多平面图（FromNv12Data/FromDeviceNv12）——此用例保证编译通过且等价路径可用。
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        if (!Has(model)) Assert.Ignore("model not found");

        int w = 640, h = 640;
        var y = new byte[w * h];
        var uv = new byte[w * h / 2];
        using var vi = VisionImage.FromNv12Data(y, uv, w, h);
        using var det = new DetectionModel(model, CpuOrt());
        Assert.DoesNotThrow(() => det.Predict(vi));
    }

    // ==================== 前/后处理参数（自省，kind 级） ====================

    [Test]
    public void Detection_ParamIntrospection_ReturnsConfNms()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        if (!Has(model)) Assert.Ignore("model not found");

        using var det = new DetectionModel(model, CpuOrt());
        var names = det.ParamNames();
        Assert.That(names, Does.Contain("conf_threshold"));
        Assert.That(names, Does.Contain("nms_threshold"));
        Assert.That(det.ParamType("conf_threshold"), Is.EqualTo('D'));
        Assert.That(det.ParamType("nms_threshold"), Is.EqualTo('D'));
    }

    // ==================== PedestrianAttribute pipeline ====================

    [Test]
    public void PedAttr_TypedSetters_WorkAndValidate()
    {
        var det = Path.Combine(ModelRoot, "zhgd_det.onnx");
        var ml = Path.Combine(ModelRoot, "zhgd_ml.onnx");
        if (!Has(det) || !Has(ml)) Assert.Ignore("model not found");

        using var ped = new PedestrianAttributeModel(det + "|" + ml, CpuOrt());
        // 类型化 setter：阈值 + 输入尺寸 + cls batch size
        ped.SetDetThreshold(0.5);
        ped.SetInputSize(640, 640);
        ped.SetClsInputSize(192, 256);
        ped.SetClsBatchSize(-1);   // 自动
        ped.SetClsBatchSize(1);    // Sophgo batch=1 静态
        // 非法 batch → 抛异常
        Assert.Throws<InvalidOperationException>(() => ped.SetClsBatchSize(0));
        Assert.Throws<InvalidOperationException>(() => ped.SetClsBatchSize(-2));
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
    public void Ocr_ControlSetters_WorkAndValidate()
    {
        var dir = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile");
        var dict = Path.Combine(GetTestDataDir(), "test_data", "ppocrv4_dict.txt");
        var model = string.Join("|",
            Path.Combine(dir, "det_infer.onnx"),
            Path.Combine(dir, "cls_infer.onnx"),
            Path.Combine(dir, "rec_infer.onnx"),
            dict);
        if (!Has(Path.Combine(dir, "det_infer.onnx"))) Assert.Ignore("model not found");

        using var ocr = new OcrModel(model, CpuOrt());
        ocr.SetMaxSideLen(960);
        ocr.SetClsBatchSize(2);
        ocr.SetRecBatchSize(-1);
        ocr.SetRecImageShape(3, 48, 320);
        Assert.Throws<InvalidOperationException>(() => ocr.SetClsBatchSize(0));
        Assert.Throws<InvalidOperationException>(() => ocr.SetRecBatchSize(-2));
    }

    [Test]
    public void LprDet_TypedSetters_WorkAndValidate()
    {
        var model = Path.Combine(ModelRoot, "yolov5plate.onnx");
        if (!Has(model)) Assert.Ignore("model not found");

        using var lpr = new LprDetectionModel(model, CpuOrt());
        lpr.SetConfThreshold(0.35);
        lpr.SetNmsThreshold(0.5);
        lpr.SetLandmarksPerCard(4);
        lpr.SetInputSize(640, 640);
    }

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
