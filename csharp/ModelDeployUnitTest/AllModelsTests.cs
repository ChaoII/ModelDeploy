using ModelDeploy;
using ModelDeploy.Models;
using NUnit.Framework;

namespace ModelDeployUnitTest;

/// <summary>
/// 覆盖全部模型 kind 的端到端验证（真实模型 + 测试数据）。
/// 任一 kind 无法推理 = 重构回归。
/// </summary>
public class AllModelsTests
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

    [Test]
    public void Detection_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new DetectionModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].Score, Is.GreaterThan(0));
    }

    [Test]
    public void Detection_DrawResult_IsOptional_AndSimple()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var canvas = vi.Clone();
        using var m = new DetectionModel(model, CpuOrt());

        // 场景1：纯推理，不 draw（结果可枚举读取；句柄随 using 释放）
        using var pure = m.Predict(vi);
        Assert.That(pure, Is.Not.Empty);

        // 场景2：需要可视化时，句柄直达 result.Draw（复用 C++ vis_det）
        pure.Draw(canvas, new DrawOptions
        {
            Threshold = 0.4,
            LabelMap = new Dictionary<int, string> { { 0, "person" }, { 1, "bicycle" }, { 2, "car" } },
            FontSize = 16,
            Alpha = 0.3,
        });

        var tmp = Path.Combine(Path.GetTempPath(), "md_predraw_test.png");
        try
        {
            canvas.Save(tmp);
            Assert.That(new FileInfo(tmp).Length, Is.GreaterThan(0));
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Test]
    public void Detection_Clone_IndependentInstance()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new DetectionModel(model, CpuOrt());
        using var clone = m.Clone();
        Assert.That(clone.IsReady(), Is.True);
        // 原模型与克隆各自独立推理，结果一致
        var a = m.Predict(vi);
        var b = clone.Predict(vi);
        Assert.That(a.Count, Is.EqualTo(b.Count));
        Assert.That(a[0].Score, Is.EqualTo(b[0].Score).Within(1e-4f));
    }

    [Test]
    public void Classification_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-cls.onnx");
        var img = Path.Combine(ImageRoot, "bus.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new ClassificationModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].LabelId, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void Pose_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-pose.onnx");
        var img = Path.Combine(ImageRoot, "bus.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new PoseModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].KeyPoints.Length, Is.GreaterThan(0));
    }

    [Test]
    public void Obb_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-obb.onnx");
        var img = Path.Combine(ImageRoot, "bus.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new ObbModel(model, CpuOrt());
        // 不强制有检测（取决于图），只验证不崩溃
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Null);
    }

    [Test]
    public void InstanceSeg_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-seg.onnx");
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new InstanceSegModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].Mask.Length, Is.GreaterThan(0));
    }

    [Test]
    public void SemSeg_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo26n", "yolo26n-sem.onnx");
        var img = Path.Combine(ImageRoot, "bus.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new SemSegModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r[0].Labels.Length, Is.GreaterThan(0));
        Assert.That(r[0].NumClasses, Is.GreaterThan(0));
    }

    [Test]
    public void Depth_Works()
    {
        var model = Path.Combine(ModelRoot, "yolo26n", "yolo26n-depth.onnx");
        var img = Path.Combine(ImageRoot, "bus.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new DepthModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r[0].Depth.Length, Is.GreaterThan(0));
    }

    [Test]
    public void FaceDet_Works()
    {
        var model = Path.Combine(ModelRoot, "face", "scrfd_2.5g_bnkps_shape640x640.onnx");
        var img = Path.Combine(ImageRoot, "test_face_detection4.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new FaceDetModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].KeyPoints.Length, Is.GreaterThan(0));
    }

    [Test]
    public void FaceRec_Works()
    {
        var model = Path.Combine(ModelRoot, "face", "face_recognizer.onnx");
        var img = Path.Combine(ImageRoot, "test_face_id.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new FaceRecModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r.Embedding.Length, Is.GreaterThan(0));
    }
    [Test]
    public void ReId_Works()
    {
        var model = Path.Combine(ModelRoot, "osnet_x1_0.onnx");
        var img = Path.Combine(ImageRoot, "test_face_id.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new ReIdModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r.Embedding.Length, Is.GreaterThan(0));
    }

    [Test]
    public void FaceAge_Works()
    {
        var model = Path.Combine(ModelRoot, "face", "age_predictor.onnx");
        var img = Path.Combine(ImageRoot, "test_face_id1.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new FaceAgeModel(model, CpuOrt());
        var age = m.Predict(vi);
        Assert.That(age, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void FaceGender_Works()
    {
        var model = Path.Combine(ModelRoot, "face", "gender_predictor.onnx");
        var img = Path.Combine(ImageRoot, "test_face_gender.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new FaceGenderModel(model, CpuOrt());
        var gender = m.Predict(vi);
        Assert.That(gender, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void InsightFace_Works()
    {
        var dir = Path.Combine(ModelRoot, "insightface", "buffalo_l");
        var model = string.Join("|",
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            Path.Combine(dir, "genderage.onnx"));
        var img = Path.Combine(ImageRoot, "test_face1.jpg");
        if (!Has(Path.Combine(dir, "det_10g.onnx")) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new InsightFaceModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].Embedding.Length, Is.GreaterThan(0));
        Assert.That(r[0].Gender, Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void InsightFace_Clone_IndependentInstance()
    {
        var dir = Path.Combine(ModelRoot, "insightface", "buffalo_l");
        var model = string.Join("|",
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            Path.Combine(dir, "genderage.onnx"));
        var img = Path.Combine(ImageRoot, "test_face1.jpg");
        if (!Has(Path.Combine(dir, "det_10g.onnx")) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new InsightFaceModel(model, CpuOrt());
        using var clone = m.Clone();
        Assert.That(clone.IsReady(), Is.True);
        var a = m.Predict(vi);
        var b = clone.Predict(vi);
        Assert.That(a.Count, Is.EqualTo(b.Count));
        Assert.That(a[0].Embedding.Length, Is.EqualTo(b[0].Embedding.Length));
    }

    [Test]
    public void Ocr_Works()
    {
        var dir = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile");
        var dict = Path.Combine(GetTestDataDir(), "test_data", "ppocrv4_dict.txt");
        var model = string.Join("|",
            Path.Combine(dir, "det_infer.onnx"),
            Path.Combine(dir, "cls_infer.onnx"),
            Path.Combine(dir, "rec_infer.onnx"),
            dict);
        var img = Path.Combine(ImageRoot, "test_ocr.png");
        if (!Has(Path.Combine(dir, "det_infer.onnx")) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new OcrModel(model, CpuOrt());
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Empty);
        Assert.That(r[0].Text, Is.Not.Empty);
    }

    [Test]
    public void Lpr_Works()
    {
        var det = Path.Combine(ModelRoot, "yolov5plate.onnx");
        var rec = Path.Combine(ModelRoot, "plate_recognition_color.onnx");
        var model = $"{det}|{rec}";
        var img = Path.Combine(ImageRoot, "test_lpr_pipeline2.jpg");
        if (!Has(det) || !Has(rec) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new LprModel(model, CpuOrt());
        // 不强制检测到车牌，只验证不崩溃且结果结构正常
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Null);
    }

    [Test]
    public void PedestrianAttribute_Works()
    {
        var det = Path.Combine(ModelRoot, "zhgd_det.onnx");
        var cls = Path.Combine(ModelRoot, "zhgd_ml.onnx");
        var model = $"{det}|{cls}";
        var img = Path.Combine(ImageRoot, "test_pedestrian_attribute1.jpg");
        if (!Has(det) || !Has(cls) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new PedestrianAttributeModel(model, CpuOrt());
        m.SetInputSize(1280, 1280);    // zhgd_det.onnx 输入 1280x1280
        m.SetClsInputSize(192, 256);   // zhgd_ml.onnx 输入 192(W)x256(H)
        var r = m.Predict(vi);
        Assert.That(r, Is.Not.Null);
    }

    /// <summary>各模型 DrawResult 可绘制（复用 C++ vis_*）。</summary>
    [Test]
    public void DrawResult_AllModels_Work()
    {
        var img = Path.Combine(ImageRoot, "test_detection0.jpg");
        if (!Has(img)) Assert.Ignore("image not found");
        using var vi = VisionImage.Read(img);

        // Detection
        {
            var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n.onnx");
            if (Has(model))
            {
                using var m = new DetectionModel(model, CpuOrt());
                using var results = m.Predict(vi);
                using var canvas = vi.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions { Threshold = 0.4 }));
            }
        }
        // Pose
        {
            var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-pose.onnx");
            var img2 = Path.Combine(ImageRoot, "bus.jpg");
            if (Has(model) && Has(img2))
            {
                using var vi2 = VisionImage.Read(img2);
                using var m = new PoseModel(model, CpuOrt());
                using var results = m.Predict(vi2);
                using var canvas = vi2.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions()));
            }
        }
        // OBB
        {
            var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-obb.onnx");
            var img2 = Path.Combine(ImageRoot, "bus.jpg");
            if (Has(model) && Has(img2))
            {
                using var vi2 = VisionImage.Read(img2);
                using var m = new ObbModel(model, CpuOrt());
                using var results = m.Predict(vi2);
                using var canvas = vi2.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions()));
            }
        }
        // InstanceSeg
        {
            var model = Path.Combine(ModelRoot, "yolo11n", "yolo11n-seg.onnx");
            if (Has(model))
            {
                using var m = new InstanceSegModel(model, CpuOrt());
                using var results = m.Predict(vi);
                using var canvas = vi.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions { Threshold = 0.4 }));
            }
        }
        // SemSeg
        {
            var model = Path.Combine(ModelRoot, "yolo26n", "yolo26n-sem.onnx");
            if (Has(model))
            {
                using var m = new SemSegModel(model, CpuOrt());
                using var r = m.Predict(vi);
                using var canvas = vi.Clone();
                Assert.DoesNotThrow(() => r.Draw(canvas, new DrawOptions { Alpha = 0.5 }));
            }
        }
        // Depth
        {
            var model = Path.Combine(ModelRoot, "yolo26n", "yolo26n-depth.onnx");
            if (Has(model))
            {
                using var m = new DepthModel(model, CpuOrt());
                using var r = m.Predict(vi);
                using var canvas = vi.Clone();
                Assert.DoesNotThrow(() => r.Draw(canvas, new DrawOptions()));
            }
        }
        // OCR
        {
            var dir = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile");
            var dict = Path.Combine(GetTestDataDir(), "test_data", "ppocrv4_dict.txt");
            var model = string.Join("|",
                Path.Combine(dir, "det_infer.onnx"),
                Path.Combine(dir, "cls_infer.onnx"),
                Path.Combine(dir, "rec_infer.onnx"),
                dict);
            var img2 = Path.Combine(ImageRoot, "test_ocr.png");
            if (Has(Path.Combine(dir, "det_infer.onnx")) && Has(img2))
            {
                using var vi2 = VisionImage.Read(img2);
                using var m = new OcrModel(model, CpuOrt());
                using var results = m.Predict(vi2);
                using var canvas = vi2.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions()));
            }
        }
        // LPR
        {
            var det = Path.Combine(ModelRoot, "yolov5plate.onnx");
            var rec = Path.Combine(ModelRoot, "plate_recognition_color.onnx");
            var model = $"{det}|{rec}";
            var img2 = Path.Combine(ImageRoot, "test_lpr_pipeline2.jpg");
            if (Has(det) && Has(rec) && Has(img2))
            {
                using var vi2 = VisionImage.Read(img2);
                using var m = new LprModel(model, CpuOrt());
                using var results = m.Predict(vi2);
                using var canvas = vi2.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions()));
            }
        }
        // FaceDet
        {
            var model = Path.Combine(ModelRoot, "face", "scrfd_2.5g_bnkps_shape640x640.onnx");
            var img2 = Path.Combine(ImageRoot, "test_face_detection4.jpg");
            if (Has(model) && Has(img2))
            {
                using var vi2 = VisionImage.Read(img2);
                using var m = new FaceDetModel(model, CpuOrt());
                using var results = m.Predict(vi2);
                using var canvas = vi2.Clone();
                Assert.DoesNotThrow(() => results.Draw(canvas, new DrawOptions()));
            }
        }
    }

    [Test]
    public void Asr_Works()
    {
        var dir = Path.Combine(ModelRoot, "sense_voice");
        var model = string.Join("|",
            Path.Combine(dir, "model.int8.onnx"),
            Path.Combine(dir, "tokens.txt"));
        var wav = Path.Combine(dir, "test_wavs", "zh.wav");
        if (!Has(Path.Combine(dir, "model.int8.onnx")) || !Has(wav)) Assert.Ignore("data missing");
        using var m = new SenseVoiceModel(model, CpuOrt());
        var r = m.PredictWav(wav);
        Assert.That(r.Text, Is.Not.Empty);
    }

    [Test]
    public void Tts_ModelLoads()
    {
        // Kokoro predict 需 s2t_map.bin/t2s_map.bin（测试数据未携带），此处验证模型可加载。
        var dir = Path.Combine(ModelRoot, "kokoro_v1_1");
        var model = string.Join("|",
            Path.Combine(dir, "model.onnx"),
            Path.Combine(dir, "tokens.txt"),
            Path.Combine(dir, "lexicon-gb-en.txt"),
            Path.Combine(dir, "lexicon-zh.txt"),
            Path.Combine(dir, "voices.bin"),
            Path.Combine(dir, "dict"),
            dir);
        if (!Has(Path.Combine(dir, "model.onnx"))) Assert.Ignore("data missing");
        using var m = new KokoroModel(model, CpuOrt());
        Assert.That(m.IsReady(), Is.True);
    }

    // ═══════════════════ 子模型测试 ═══════════════════

    [Test]
    public void OcrDet_Works()
    {
        var model = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile", "det_infer.onnx");
        var img = Path.Combine(ImageRoot, "test_ocr.png");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new DbDetectorModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r.Count, Is.GreaterThan(0));
    }

    [Test]
    public void OcrRec_Works()
    {
        var dir = Path.Combine(ModelRoot, "ocr", "ppocrv4_mobile");
        var model = $"{Path.Combine(dir, "rec_infer.onnx")}|{Path.Combine(GetTestDataDir(), "test_data", "ppocrv4_dict.txt")}";
        var img = Path.Combine(ImageRoot, "test_ocr_recognition1.jpg");
        if (!Has(Path.Combine(dir, "rec_infer.onnx")) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new RecognizerModel(model, CpuOrt());
        using var r = m.Predict(vi);
        // rec 模型依赖输入为单行文本裁剪图，识别结果可能为空；验证流程不崩溃
        Assert.That(r, Is.Not.Null);
    }

    [Test]
    public void FaceRecPipeline_Works()
    {
        var dir = Path.Combine(ModelRoot, "face");
        var model = $"{Path.Combine(dir, "scrfd_2.5g_bnkps_shape640x640.onnx")}|{Path.Combine(dir, "face_recognizer.onnx")}";
        var img = Path.Combine(ImageRoot, "test_face_detection4.jpg");
        if (!Has(Path.Combine(dir, "scrfd_2.5g_bnkps_shape640x640.onnx")) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new FaceRecognizerPipelineModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r.Count, Is.GreaterThan(0));
        Assert.That(r[0].Embedding.Length, Is.GreaterThan(0));
    }

    [Test]
    public void FaceGenderAge_Works()
    {
        // 年龄/性别是 face/ 下的两个独立 Seeta 子模型（非 insightface genderage）。
        var ageModel = Path.Combine(ModelRoot, "face", "age_predictor.onnx");
        var genderModel = Path.Combine(ModelRoot, "face", "gender_predictor.onnx");
        var img = Path.Combine(ImageRoot, "test_face1.jpg");
        if (!Has(ageModel) || !Has(genderModel) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var ageM = new FaceAgeModel(ageModel, CpuOrt());
        using var genderM = new FaceGenderModel(genderModel, CpuOrt());
        Assert.That(ageM.Predict(vi), Is.GreaterThanOrEqualTo(0));
        Assert.That(genderM.Predict(vi), Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void InsightFaceDet_Works()
    {
        var model = Path.Combine(ModelRoot, "insightface", "buffalo_l", "det_10g.onnx");
        var img = Path.Combine(ImageRoot, "test_face1.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new InsightFaceDetModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r.Count, Is.GreaterThan(0));
        Assert.That(r[0].KeyPoints.Length, Is.GreaterThan(0));
    }

    [Test]
    public void LprDet_Works()
    {
        var model = Path.Combine(ModelRoot, "yolov5plate.onnx");
        var img = Path.Combine(ImageRoot, "test_lpr_pipeline2.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new LprDetectionModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r, Is.Not.Null);
    }

    [Test]
    public void LprRec_Works()
    {
        var model = Path.Combine(ModelRoot, "plate_recognition_color.onnx");
        var img = Path.Combine(ImageRoot, "test_lpr_recognizer.jpg");
        if (!Has(model) || !Has(img)) Assert.Ignore("data missing");
        using var vi = VisionImage.Read(img);
        using var m = new LprRecognizerModel(model, CpuOrt());
        using var r = m.Predict(vi);
        Assert.That(r.Count, Is.GreaterThan(0));
    }
}
