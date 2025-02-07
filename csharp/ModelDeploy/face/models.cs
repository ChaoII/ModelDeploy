using System.Runtime.InteropServices;
using ModelDeploy.detection;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.face;

public class SeetaFace
{
    private MDModel _model;

    public SeetaFace(string modelDir, int moduleFlags = FaceConstants.MD_MASK, int threadNum = 8)
    {
        _model = new MDModel();
        int ret = md_create_face_model(ref _model, modelDir, moduleFlags, threadNum);
        if (ret != 0)
        {
            throw new Exception("Failed to create face model,error code: " + ret);
        }
    }

    ~SeetaFace()
    {
        md_free_face_model(ref _model);
    }

    public List<DetectionResult> FaceDetect(Image image)
    {
        var result = new MDDetectionResults();
        int ret = md_face_detection(ref _model, ref image.RawImage, ref result);
        if (ret != 0)
        {
            throw new Exception("md_face_detection failed error code is: " + ret);
        }

        var results = DetectionResult.FromMDDetectionResults(result);
        // todo optimize structure
        YOLOv8.md_free_detection_result(ref result);
        return results;
    }

    public LandMarkResult FaceLandmark(Image image, Rect rect)
    {
        var cResult = new MDLandMarkResult();
        var cRect = rect.ToRaw();
        int ret = md_face_marker(ref _model, ref image.RawImage, ref cRect, ref cResult);
        if (ret != 0)
        {
            throw new Exception("md_face_marker failed error code is: " + ret);
        }

        var result = LandMarkResult.FromRow(cResult);
        md_free_face_landmark(ref cResult);
        return result;
    }

    public FaceFeature FaceFeatureExtract(Image image, LandMarkResult points)
    {
        var cFeature = new MDFaceFeature();
        MDLandMarkResult cPoints = LandMarkResult.ToRow(points);
        int ret = md_face_feature(ref _model, ref image.RawImage, ref cPoints, ref cFeature);
        if (ret != 0)
        {
            throw new Exception("md_face_feature failed error code is: " + ret);
        }

        var feature = FaceFeature.FromRow(cFeature);
        md_free_face_feature(ref cFeature);
        return feature;
    }

    public FaceFeature FaceFeatureExtract(Image image)
    {
        var cFeature = new MDFaceFeature();
        int ret = md_face_feature_e2e(ref _model, ref image.RawImage, ref cFeature);
        if (ret != 0)
        {
            throw new Exception("md_face_feature_e2e failed error code is: " + ret);
        }

        var feature = FaceFeature.FromRow(cFeature);
        md_free_face_feature(ref cFeature);
        return feature;
    }

    public float FaceFeatureCompare(FaceFeature feature1, FaceFeature feature2)
    {
        var similarity = 0.0f;
        var cFeature1 = FaceFeature.ToRow(feature1);
        var cFeature2 = FaceFeature.ToRow(feature2);
        int ret = md_face_feature_compare(ref _model, ref cFeature1, ref cFeature2, ref similarity);
        if (ret != 0)
        {
            throw new Exception("md_face_feature_compare failed error code is: " + ret);
        }

        return similarity;
    }

    public FaceAntiSpoofingResult FaceAntiSpoofing(Image image)
    {
        var result = new FaceAntiSpoofingResult();
        int ret = md_face_anti_spoofing(ref _model, ref image.RawImage, ref result);
        if (ret != 0)
        {
            throw new Exception("md_face_anti_spoofing failed error code is: " + ret);
        }

        return result;
    }

    public FaceQualityEvaluateResult FaceQualityEvaluate(Image image, FaceQualityEvaluateType type)
    {
        var result = new FaceQualityEvaluateResult();
        int ret = md_face_quality_evaluate(ref _model, ref image.RawImage, type, ref result);
        if (ret != 0)
        {
            throw new Exception("md_face_quality_evaluate failed error code is: " + ret);
        }

        return result;
    }

    public int FaceAgePredict(Image image)
    {
        var age = 0;
        int ret = md_face_age_predict(ref _model, ref image.RawImage, ref age);
        if (ret != 0)
        {
            throw new Exception("md_face_age_predict failed error code is: " + ret);
        }

        return age;
    }

    public GenderResult FaceGenderPredict(Image image)
    {
        var gender = new GenderResult();
        int ret = md_face_gender_predict(ref _model, ref image.RawImage, ref gender);
        if (ret != 0)
        {
            throw new Exception("md_face_gender_predict failed error code is: " + ret);
        }

        return gender;
    }

    public EyeStateResult FaceEyeStatePredict(Image image)
    {
        var eyeState = new MDEyeStateResult();
        int ret = md_face_eye_state_predict(ref _model, ref image.RawImage, ref eyeState);
        if (ret != 0)
        {
            throw new Exception("md_face_eye_state_predict failed error code is: " + ret);
        }

        return EyeStateResult.FromRow(eyeState);
    }


    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int
        md_create_face_model(ref MDModel model, string modelDir, int flags = FaceConstants.MD_MASK, int threadNum = 1);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_detection(ref MDModel model, ref MDImage image, ref MDDetectionResults result);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int
        md_face_marker(ref MDModel model, ref MDImage image, ref MDRect rect, ref MDLandMarkResult result);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_feature(ref MDModel model, ref MDImage image,
        ref MDLandMarkResult points, ref MDFaceFeature feature);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_feature_e2e(ref MDModel model, ref MDImage image, ref MDFaceFeature feature);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_feature_compare(ref MDModel model, ref MDFaceFeature feature1,
        ref MDFaceFeature feature2, ref float similarity);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_anti_spoofing(ref MDModel model, ref MDImage image,
        ref FaceAntiSpoofingResult result);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_quality_evaluate(ref MDModel model, ref MDImage image,
        FaceQualityEvaluateType type, ref FaceQualityEvaluateResult result);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_age_predict(ref MDModel model, ref MDImage image, ref int age);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_gender_predict(ref MDModel model, ref MDImage image, ref GenderResult gender);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int md_face_eye_state_predict(ref MDModel model, ref MDImage image,
        ref MDEyeStateResult eyeState);


    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void md_free_face_landmark(ref MDLandMarkResult result);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void md_free_face_feature(ref MDFaceFeature feature);

    [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void md_free_face_model(ref MDModel model);
}