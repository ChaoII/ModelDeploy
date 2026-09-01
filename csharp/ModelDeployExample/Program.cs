using System.Collections.Generic;
using ModelDeploy;
using ModelDeploy.Models;

namespace TestModelDeploy;

static class Program
{
    /// <summary>仓库根（从 exe 输出目录 net9.0 上溯 5 级），保证从任何工作目录运行都能找到测试数据。</summary>
    private static readonly string TestDataPath =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));

    private static RuntimeOption CpuOrt() =>
        new RuntimeOption().UseOrt().SetDevice(Device.CPU).SetCpuThreads(4);

    static void TestImage()
    {
        // 读图
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/test_detection0.jpg"));
        Console.WriteLine($"image: {image.Width}x{image.Height}");

        // 绘制：矩形 + 多边形 + 文本（就地修改图像）
        Draw.DrawRect(image, 100, 100, 200, 200, 255, 0, 0, 0.5f);
        Draw.DrawPolygon(image, new float[] { 300, 300, 380, 320, 350, 400, 290, 380 },
            new float[] { 300, 300, 320, 380, 400, 350, 380, 290 }, 0, 255, 0, 0.5f);
        Draw.DrawText(image, 100, 340, "Hello ModelDeploy", null, 24, 255, 255, 255);

        // 克隆 + 裁剪
        using var clone = image.Clone();
        using var crop = image.Crop(50, 50, 200, 150);
        Console.WriteLine($"crop: {crop.Width}x{crop.Height}");

        // 编码
        var jpg = crop.Encode(".jpg");
        Console.WriteLine($"encoded {jpg.Length} bytes (JPEG magic: {jpg[0]:X2}{jpg[1]:X2})");

        // 保存
        image.Save("annotated.jpg");
        Console.WriteLine("saved annotated.jpg");

        // 显示（阻塞：等待用户在窗口按键后关闭；终端/CI 环境请保持注释）
        // 若你的 OpenCV 未编译 HighGUI，Show() 会抛出 InvalidOperationException。
        // crop.Show();
        Console.WriteLine("show skipped (OpenCV HighGUI 阻塞窗口，终端环境默认注释)；取消注释可查看窗口");
    }

    static void TestDetection()
    {
        using var det = new DetectionModel(
            Path.Combine(TestDataPath, "test_models/onnx/yolo11n/yolo11n.onnx"), CpuOrt());
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/test_detection0.jpg"));

        // 演示参数 API：类型化 setter（编译期检查，杜绝字符串拼错）+ 参数名/类型自省
        det.SetConfThreshold(0.4);
        det.SetNmsThreshold(0.45);
        Console.WriteLine("params: " + string.Join(", ", det.ParamNames())
                                     + " | conf_threshold type=" + det.ParamType("conf_threshold")
                                     + " nms_threshold type=" + det.ParamType("nms_threshold"));

        // 纯推理：Prediction<T> 同时是可枚举的列表（读结果）
        using var result = det.Predict(image);
        Console.WriteLine($"detected {result.Count} objects");
        foreach (var r in result)
            Console.WriteLine($"  {r}");

        // 需要可视化时：result.Draw，句柄直达 C++ vis_det（可控制阈值/类别名/字体/透明度）
        // using var canvas = image.Clone();
        result.Draw(image, new DrawOptions
        {
            Threshold = 0.4,
            LabelMap = new Dictionary<int, string> { { 0, "person" }, { 1, "bicycle" }, { 2, "car" } },
            FontSize = 16,
            Alpha = 0.3,
        });
        image.Save("detection_annotated.jpg");
        image.Show();
        Console.WriteLine("visualized -> detection_annotated.jpg");
    }

    static void TestClassification()
    {
        using var cls = new ClassificationModel(
            Path.Combine(TestDataPath, "test_models/onnx/yolo11n/yolo11n-cls.onnx"), CpuOrt());
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/bus.jpg"));
        using var clsResult = cls.Predict(image);
        foreach (var r in clsResult)
            Console.WriteLine(r);
    }

    static void TestPose()
    {
        using var pose = new PoseModel(
            Path.Combine(TestDataPath, "test_models/onnx/yolo11n/yolo11n-pose.onnx"), CpuOrt());
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/bus.jpg"));
        using var poseResult = pose.Predict(image);
        foreach (var r in poseResult)
            Console.WriteLine(r);
    }

    static void TestOCR()
    {
        var dir = Path.Combine(TestDataPath, "test_models/onnx/ocr/ppocrv4_mobile");
        var model = string.Join("|",
            Path.Combine(dir, "det_infer.onnx"),
            Path.Combine(dir, "cls_infer.onnx"),
            Path.Combine(dir, "rec_infer.onnx"),
            Path.Combine(TestDataPath, "ppocrv4_dict.txt"));
        using var ocr = new OcrModel(model, CpuOrt());
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/test_ocr.png"));
        using var ocrResult = ocr.Predict(image);
        foreach (var r in ocrResult)
            Console.WriteLine(r);
    }

    static void TestInsightFace()
    {
        var dir = Path.Combine(TestDataPath, "test_models/onnx/insightface/buffalo_l");
        var model = string.Join("|",
            Path.Combine(dir, "det_10g.onnx"),
            Path.Combine(dir, "w600k_r50.onnx"),
            Path.Combine(dir, "2d106det.onnx"),
            Path.Combine(dir, "1k3d68.onnx"),
            Path.Combine(dir, "genderage.onnx"));
        using var face = new InsightFaceModel(model, CpuOrt());
        using var image = VisionImage.Read(Path.Combine(TestDataPath, "test_images/test_face1.jpg"));
        using var faceResult = face.Predict(image);
        foreach (var r in faceResult)
            Console.WriteLine(r);
    }

    static void TestPedestrianAttribute()
    {
        static RuntimeOption CpuOrt() =>
            new RuntimeOption().UseOrt().SetDevice(Device.CPU).SetCpuThreads(4);

        // var opt = new RuntimeOption();
        // opt.UseOrt();

        var dir = Path.Combine(TestDataPath, "test_models/onnx");
        // 两参重载：det 模型路径 + 分类模型路径（内部拼成 capi 的 "det|cls"；onnx/engine/mnn 皆可）
        using var ped = new PedestrianAttributeModel(
            Path.Combine(dir, "zhgd_det.onnx"),
            Path.Combine(dir, "zhgd_ml.onnx"),
            CpuOrt());
        ped.SetInputSize(1280, 1280);
        ped.SetClsInputSize(192, 256);
        ped.SetClsBatchSize(8); // >0 固定 / -1 自动；Sophgo batch=1 静态
        ped.SetDetThreshold(0.25);
        using var image = VisionImage.Read(
            Path.Combine(TestDataPath, "test_images/test_pedestrian_attribute_scale.jpg"));
        using var result = ped.Predict(image); // Prediction<AttributeResult>
        Console.WriteLine($"persons: {result.Count}");
        foreach (var r in result)
            Console.WriteLine($"  {r.Box} score={r.BoxScore:F3} attrs=[{string.Join(", ", r.AttrScores)}]");
        // 绘制：句柄直达 C++ vis_attr（框 + 属性文本），就地在 image 上画

        Dictionary<int, string> dict = new Dictionary<int, string>
            { { 0, "傻话" }, { 1, "雷达" }, { 2, "哈拉" }, { 3, "糍粑" }, { 4, "索拉" } };

        result.Draw(image,
            new DrawOptions
            {
                Threshold = 0.25, FontSize = 10, Alpha = 0.15, LabelMap = dict, AbnormalIds = new List<int> { 0, 1 },
                ShowAttr = false
            });
        image.Save("pedattr_annotated.jpg");
        image.Show();
        Console.WriteLine("visualized -> pedattr_annotated.jpg");
    }

    static void TestSenseVoice()
    {
        var dir = Path.Combine(TestDataPath, "test_models/onnx/sense_voice");
        var model = string.Join("|",
            Path.Combine(dir, "model.int8.onnx"),
            Path.Combine(dir, "tokens.txt"));
        using var asr = new SenseVoiceModel(model, CpuOrt());
        var result = asr.PredictWav(Path.Combine(dir, "test_wavs", "zh.wav"));
        Console.WriteLine($"ASR: {result.Text}");
    }

    static void TestKokoro()
    {
        var dir = Path.Combine(TestDataPath, "test_models/onnx/kokoro_v1_1");
        var model = string.Join("|",
            Path.Combine(dir, "model.onnx"),
            Path.Combine(dir, "tokens.txt"),
            Path.Combine(dir, "lexicon-gb-en.txt"),
            Path.Combine(dir, "lexicon-zh.txt"),
            Path.Combine(dir, "voices.bin"),
            Path.Combine(dir, "dict"),
            TestDataPath);
        using var tts = new KokoroModel(model, CpuOrt());
        var result = tts.Predict("你好世界今天天气不错 hello world", "zf_001", 1.0f);
        tts.SaveWav(result, "output.wav");
        Console.WriteLine($"TTS: {result.Audio.Length} samples @ {result.SampleRate}Hz -> output.wav");
    }

    static void Main(string[] args)
    {
        // TestDetection();
        TestPedestrianAttribute();
        // TestImage();
        // TestClassification();
        // TestPose();
        // TestOCR();
        // TestInsightFace();
        // TestSenseVoice();
        // TestKokoro();
    }
}