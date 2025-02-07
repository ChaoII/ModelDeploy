using ModelDeploy;
using ModelDeploy.asr;
using ModelDeploy.detection;
using ModelDeploy.face;
using ModelDeploy.ocr;
using ModelDeploy.types_internal_c;
using ModelDeploy.utils;

namespace ConsoleApp1;

class Program
{
    static void TestASR()
    {
        FunASROffLine funASR = new FunASROffLine(
            "D:/funasr-runtime-resources/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
            "D:/funasr-runtime-resources/models/speech_fsmn_vad_zh-cn-16k-common-onnx",
            "D:/funasr-runtime-resources/models/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
            "D:/funasr-runtime-resources/models/fst_itn_zh",
            "D:/funasr-runtime-resources/models/speech_ngram_lm_zh-cn-ai-wesp-fst");
        ASRResult result = funASR.Predict("D:/funasr-runtime-resources/vad_example.wav");
        Console.WriteLine(result.Message);
    }

    static void TestDetection()
    {
        Image image = Image.Read("D:/funasr-runtime-resources/test_detection.png");
        YOLOv8 yolov8 = new YOLOv8("D:/funasr-runtime-resources/best.onnx");
        yolov8.SetInputSize(1440, 1440);
        List<DetectionResult> detectionResults = yolov8.Predict(image);
        yolov8.DrawDetectionResult(image, detectionResults, "D:/funasr-runtime-resources/msyh.ttc", 20, 0.5);
        image.Show();
        foreach (var detectionResult in detectionResults)
        {
            Console.WriteLine(detectionResult.Box);
        }
    }

    static void TestImage()
    {
        Image image = Image.Read("D:/funasr-runtime-resources/test_detection.png");
        MDRect rect = new MDRect { x = 0, y = 0, width = 100, height = 100 };
        MDRect rect1 = new MDRect { x = 200, y = 200, width = 100, height = 100 };
        MDRect rect2 = new MDRect { x = 100, y = 100, width = 500, height = 500 };
        MDColor color = new MDColor { r = 255, g = 0, b = 0 };
        draw.DrawRect(image, rect, color, 0.5);
        draw.DrawText(image, rect1, "Hello World", "D:/funasr-runtime-resources/msyh.ttc", 20, color, 0.5);
        var cloneImage = image.Clone();
        var cropImage = cloneImage.Crop(rect2);
        cropImage.Show();
    }

    static void TestOCR()
    {
        MDOCRModelParameters parameters = new MDOCRModelParameters
        {
            model_dir = "E:/CLionProjects/wrzs_hys/tests/models/ocr",
            dict_path = "E:/CLionProjects/wrzs_hys/tests/key.txt",
            thread_num = 8,
            format = MDModelFormat.PaddlePaddle,
            max_side_len = 960,
            det_db_thresh = 0.3,
            det_db_box_thresh = 0.6,
            det_db_unclip_ratio = 1.5,
            det_db_score_mode = "slow",
            use_dilation = 0,
            rec_batch_size = 8,
        };
        Image image = Image.Read("E:/CLionProjects/wrzs_hys/tests/test_images/test_ocr.png");

        PPOCRv4 ppocrv4 = new PPOCRv4(parameters);
        List<OCRResult> results = ppocrv4.Predict(image);

        ppocrv4.DrawOcrResult(image, results, "E:/CLionProjects/wrzs_hys/tests/msyh.ttc", new Color
        {
            R = 0,
            G = 255,
            B = 255
        }, 20, 0.3);

        image.Show();

        foreach (var result in results)
        {
            Console.WriteLine(result.Text);
        }
    }

    static void TestFace()
    {
        Image image0 = Image.Read("E:/CLionProjects/wrzs_hys/tests/test_images/test_face.jpg");
        Image image1 = Image.Read("E:/CLionProjects/wrzs_hys/tests/test_images/test_face1.jpg");
        Image image2 = Image.Read("E:/CLionProjects/wrzs_hys/tests/test_images/test_face2.jpg");
        Image image3 = Image.Read("E:/CLionProjects/wrzs_hys/tests/test_images/test_face3.jpg");

        SeetaFace seetaFace =
            new SeetaFace("E:/CLionProjects/wrzs_hys/tests/models/seetaface", FaceConstants.MD_MASK, 8);

        var detectionResults = seetaFace.FaceDetect(image3);
        Console.WriteLine($"Detection {detectionResults.Count} faces in image3");

        var points = seetaFace.FaceLandmark(image3, detectionResults[0].Box);

        var feature0 = seetaFace.FaceFeatureExtract(image0);
        var feature1 = seetaFace.FaceFeatureExtract(image1);
        var feature2 = seetaFace.FaceFeatureExtract(image2);
        var feature3 = seetaFace.FaceFeatureExtract(image3, points);

        var similarity0 = seetaFace.FaceFeatureCompare(feature0, feature1);
        Console.WriteLine($"similarity between image0 and image1 is: {similarity0}");
        var similarity1 = seetaFace.FaceFeatureCompare(feature0, feature2);
        Console.WriteLine($"similarity between image0 and image2 is: {similarity1}");

        var ret = seetaFace.FaceAntiSpoofing(image0);
        Console.WriteLine($"AntiSpoofing result: {ret}");

        var age = seetaFace.FaceAgePredict(image0);
        Console.WriteLine($"Age predict result: {age}");
        var gender = seetaFace.FaceGenderPredict(image0);
        Console.WriteLine($"Gender predict result: {gender}");

        var quality0 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.Brightness);
        Console.WriteLine($"Brightness quality: {quality0}");
        var quality1 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.Clarity);
        Console.WriteLine($"Clarity quality: {quality1}");
        var quality2 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.Integrity);
        Console.WriteLine($"Integrity quality: {quality2}");
        var quality3 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.Pose);
        Console.WriteLine($"Pose quality: {quality3}");
        var quality4 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.ClarityEx);
        Console.WriteLine($"ClarityEx quality: {quality4}");
        var quality5 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.Resolution);
        Console.WriteLine($"Resolution quality: {quality5}");
        var quality6 = seetaFace.FaceQualityEvaluate(image0, FaceQualityEvaluateType.NoMask);
        Console.WriteLine($"NoMask quality: {quality6}");

        var eyeState0 = seetaFace.FaceEyeStatePredict(image0);
        Console.WriteLine($"image0: {eyeState0}");
        var eyeState1 = seetaFace.FaceEyeStatePredict(image2);
        Console.WriteLine($"image2: {eyeState1}");
    }


    static void Main(string[] args)
    {
        // TestASR();
        // TestDetection();
        // TestImage();
        // TestOCR();
        TestFace();
        int a = 30;
    }
}