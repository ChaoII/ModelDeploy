using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using ModelDeploy.V2.Results;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.V2.Models
{
    public sealed class DetectionModel : BaseModel
    {
        private DetectionModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public DetectionModel Clone() => new DetectionModel(CloneNative());

        public DetectionModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_DETECTION, modelPath, opt) { }

        public Prediction<DetectionResult> Predict(VisionImage image)
            => MakePrediction(image, ReadDetection);

        /// <summary>NV12 直接输入推理（硬解码/摄像头直通；srcDevice 指明 Y/UV 所在设备）。</summary>
        public Prediction<DetectionResult> PredictNv12(byte[] y, byte[] uv,
            int w, int h, int stepY = 0, int stepUv = 0, Device srcDevice = Device.CPU)
            => MakePredictionNv12(y, uv, w, h, stepY, stepUv, srcDevice, ReadDetection);

        /// <summary>NV12 推理并返回绑定的输入帧（设备相关的 ImageData 包装，可取平面指针/就地绘制）。
        /// Frame 为 null 或需调用方 Dispose。Frame 存活期内其引用的 y/uv 托管缓冲会被自动 pin，
        /// 保持有效且不被 GC 移动；Frame Dispose 后释放 pin。</summary>
        public (Prediction<DetectionResult> Prediction, VisionImage Frame) PredictNv12WithFrame(
            byte[] y, byte[] uv,
            int w, int h, int stepY = 0, int stepUv = 0, Device srcDevice = Device.CPU)
            => MakePredictionNv12WithFrame(y, uv, w, h, stepY, stepUv, srcDevice, ReadDetection);

        private static DetectionResult[] ReadDetection(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDDetectionItem>(result, md_result_detection);
            var list = new List<DetectionResult>(items.Length);
            foreach (var it in items)
                list.Add(new DetectionResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    LabelId = it.label_id,
                    Score = it.score
                });
            return list.ToArray();
        }
    }

    public sealed class ClassificationModel : BaseModel
    {
        private ClassificationModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public ClassificationModel Clone() => new ClassificationModel(CloneNative());

        public ClassificationModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_CLASSIFICATION, modelPath, opt) { }

        public Prediction<ClassificationResult> Predict(VisionImage image)
            => MakePrediction(image, ReadClassification);

        private static ClassificationResult[] ReadClassification(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDClassifyItem>(result, md_result_classification);
            var list = new List<ClassificationResult>(items.Length);
            foreach (var it in items)
                list.Add(new ClassificationResult { LabelId = it.label_id, Score = it.score });
            return list.ToArray();
        }
    }

    public sealed class PoseModel : BaseModel
    {
        private PoseModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public PoseModel Clone() => new PoseModel(CloneNative());

        public PoseModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_POSE, modelPath, opt) { }

        public Prediction<PoseResult> Predict(VisionImage image)
            => MakePrediction(image, ReadPose);

        private static PoseResult[] ReadPose(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDPoseItem>(result, md_result_pose);
            var list = new List<PoseResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                var kpsPtr = ResultReader.ReadItems<MDPoint3F>(result,
                    new ResultReader.ItemGetter((IntPtr h, out IntPtr k, out UIntPtr n) =>
                        md_result_keypoints(h, new UIntPtr((uint)i), out k, out n)));
                var kps = new Point3F[kpsPtr.Length];
                for (int j = 0; j < kpsPtr.Length; j++)
                    kps[j] = new Point3F(kpsPtr[j].x, kpsPtr[j].y, kpsPtr[j].z);
                list.Add(new PoseResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Score = it.score,
                    KeyPoints = kps
                });
            }
            return list.ToArray();
        }
    }

    public sealed class ObbModel : BaseModel
    {
        private ObbModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public ObbModel Clone() => new ObbModel(CloneNative());

        public ObbModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_OBB, modelPath, opt) { }

        public Prediction<ObbResult> Predict(VisionImage image)
            => MakePrediction(image, ReadObb);

        private static ObbResult[] ReadObb(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDObbItem>(result, md_result_obb);
            var list = new List<ObbResult>(items.Length);
            foreach (var it in items)
                list.Add(new ObbResult
                {
                    Box = new RotatedRectF(it.cx, it.cy, it.w, it.h, it.angle),
                    LabelId = it.label_id,
                    Score = it.score
                });
            return list.ToArray();
        }
    }

    public sealed class InstanceSegModel : BaseModel
    {
        private InstanceSegModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public InstanceSegModel Clone() => new InstanceSegModel(CloneNative());

        public InstanceSegModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_INSTANCE_SEG, modelPath, opt) { }

        public Prediction<InstanceSegResult> Predict(VisionImage image)
            => MakePrediction(image, ReadInstanceSeg);

        private static InstanceSegResult[] ReadInstanceSeg(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDIsegItem>(result, md_result_instance_seg);
            var list = new List<InstanceSegResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                md_result_mask(result, new UIntPtr((uint)i), out var maskPtr, out var mh, out var mw);
                list.Add(new InstanceSegResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    LabelId = it.label_id,
                    Score = it.score,
                    Mask = ResultReader.ReadBytes(maskPtr, new UIntPtr((ulong)mh * (ulong)mw)),
                    MaskHeight = (int)mh,
                    MaskWidth = (int)mw
                });
            }
            return list.ToArray();
        }
    }

    public sealed class SemSegModel : BaseModel
    {
        private SemSegModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public SemSegModel Clone() => new SemSegModel(CloneNative());

        public SemSegModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_SEM_SEG, modelPath, opt) { }

        public Prediction<SemSegResult> Predict(VisionImage image)
            => MakePrediction(image, ReadSemSeg);

        private static SemSegResult[] ReadSemSeg(IntPtr result)
        {
            var status = md_result_sem_seg(result, out var labels, out var h, out var w, out var numClasses);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Sem seg getter failed: {BaseModel.GetLastError()}");
            return new[]
            {
                new SemSegResult
                {
                    Labels = ResultReader.ReadBytes(labels, new UIntPtr((ulong)h * (ulong)w)),
                    Height = (int)h,
                    Width = (int)w,
                    NumClasses = numClasses
                }
            };
        }
    }

    public sealed class DepthModel : BaseModel
    {
        private DepthModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public DepthModel Clone() => new DepthModel(CloneNative());

        public DepthModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_DEPTH, modelPath, opt) { }

        public Prediction<DepthResult> Predict(VisionImage image)
            => MakePrediction(image, ReadDepth);

        private static DepthResult[] ReadDepth(IntPtr result)
        {
            var status = md_result_depth(result, out var depth, out var h, out var w);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Depth getter failed: {BaseModel.GetLastError()}");
            return new[]
            {
                new DepthResult
                {
                    Depth = ResultReader.ReadFloats(depth, new UIntPtr((ulong)h * (ulong)w)),
                    Height = (int)h,
                    Width = (int)w
                }
            };
        }
    }

    public sealed class FaceDetModel : BaseModel
    {
        private FaceDetModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public FaceDetModel Clone() => new FaceDetModel(CloneNative());

        public FaceDetModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_DET, modelPath, opt) { }

        public Prediction<FaceDetResult> Predict(VisionImage image)
            => MakePrediction(image, ReadFaceDet);

        private static FaceDetResult[] ReadFaceDet(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDFaceItem>(result, md_result_face);
            var list = new List<FaceDetResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                var kpsArr = ResultReader.ReadItems<MDPointF>(result,
                    new ResultReader.ItemGetter((IntPtr h, out IntPtr k, out UIntPtr n) =>
                        md_result_face_kps(h, new UIntPtr((uint)i), out k, out n)));
                var kps = new PointF[kpsArr.Length];
                for (int j = 0; j < kpsArr.Length; j++)
                    kps[j] = new PointF(kpsArr[j].x, kpsArr[j].y);
                list.Add(new FaceDetResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Score = it.score,
                    KeyPoints = kps
                });
            }
            return list.ToArray();
        }
    }

    public sealed class FaceRecModel : BaseModel
    {
        private FaceRecModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public FaceRecModel Clone() => new FaceRecModel(CloneNative());

        public FaceRecModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_REC, modelPath, opt) { }

        public FaceRecResult Predict(VisionImage image)
        {
            using var r = new ResultReader(PredictNative(image.Handle));
            var status = md_result_face_embedding(r.Result, UIntPtr.Zero, out var emb, out var n);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Face rec getter failed: {GetLastError()}");
            return new FaceRecResult { Embedding = ResultReader.ReadFloats(emb, n) };
        }
    }

    public sealed class FaceAgeModel : BaseModel
    {
        private FaceAgeModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public FaceAgeModel Clone() => new FaceAgeModel(CloneNative());

        public FaceAgeModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_AGE, modelPath, opt) { }

        public int Predict(VisionImage image)
        {
            using var r = new ResultReader(PredictNative(image.Handle));
            var status = md_result_age(r.Result, out var age);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Age getter failed: {GetLastError()}");
            return age;
        }
    }

    public sealed class FaceGenderModel : BaseModel
    {
        private FaceGenderModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public FaceGenderModel Clone() => new FaceGenderModel(CloneNative());

        public FaceGenderModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_GENDER, modelPath, opt) { }

        public int Predict(VisionImage image)
        {
            using var r = new ResultReader(PredictNative(image.Handle));
            var status = md_result_gender(r.Result, out var gender);
            if (status != MDStatus.MD_OK)
                throw new InvalidOperationException($"Gender getter failed: {GetLastError()}");
            return gender;
        }
    }

    public sealed class InsightFaceModel : BaseModel
    {
        private InsightFaceModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public InsightFaceModel Clone() => new InsightFaceModel(CloneNative());

        public InsightFaceModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_INSIGHTFACE, modelPath, opt) { }

        public Prediction<InsightFaceResult> Predict(VisionImage image)
            => MakePrediction(image, ReadInsightFace);

        private static InsightFaceResult[] ReadInsightFace(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDInsightFaceItem>(result, md_result_insightface);
            var list = new List<InsightFaceResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                var ui = new UIntPtr((uint)i);
                var kpsArr = ResultReader.ReadItems<MDPointF>(result,
                    new ResultReader.ItemGetter((IntPtr h, out IntPtr k, out UIntPtr n) =>
                        md_result_insightface_kps(h, ui, out k, out n)));
                md_result_insightface_embedding(result, ui, out var emb, out var embN);
                md_result_insightface_pose(result, ui, out var pose, out var poseN);
                var kps = new PointF[kpsArr.Length];
                for (int j = 0; j < kpsArr.Length; j++)
                    kps[j] = new PointF(kpsArr[j].x, kpsArr[j].y);
                list.Add(new InsightFaceResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Score = it.score,
                    KeyPoints = kps,
                    Embedding = ResultReader.ReadFloats(emb, embN),
                    Pose = ResultReader.ReadFloats(pose, poseN),
                    Gender = it.gender,
                    Age = it.age
                });
            }
            return list.ToArray();
        }
    }

    public sealed class OcrModel : BaseModel
    {
        private OcrModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public OcrModel Clone() => new OcrModel(CloneNative());

        public OcrModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_OCR, modelPath, opt) { }

        public Prediction<OcrResult> Predict(VisionImage image)
            => MakePrediction(image, ReadOcr);

        private static OcrResult[] ReadOcr(IntPtr result)
        {
            md_result_count(result, out var count);
            var list = new List<OcrResult>((int)count);
            for (uint i = 0; i < (uint)count; i++)
            {
                var status = md_result_ocr(result, new UIntPtr(i), out var quad, out var text, out var score);
                if (status != MDStatus.MD_OK) break;
                var q = new int[8];
                Marshal.Copy(quad, q, 0, 8);
                md_result_ocr_cls(result, new UIntPtr(i), out var clsLabel, out var clsScore);
                list.Add(new OcrResult
                {
                    Quad = q,
                    Text = ResultReader.ReadString(text) ?? string.Empty,
                    Score = score,
                    ClsLabel = clsLabel,
                    ClsScore = clsScore
                });
            }
            return list.ToArray();
        }
    }

    public sealed class LprModel : BaseModel
    {
        private LprModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public LprModel Clone() => new LprModel(CloneNative());

        public LprModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_LPR_PIPELINE, modelPath, opt) { }

        public Prediction<LprResult> Predict(VisionImage image)
            => MakePrediction(image, ReadLpr);

        private static LprResult[] ReadLpr(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDLprItem>(result, md_result_lpr);
            var list = new List<LprResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                md_result_plate(result, new UIntPtr((uint)i), out var plate, out var color);
                var kpsArr = ResultReader.ReadItems<MDPointF>(result,
                    new ResultReader.ItemGetter((IntPtr h, out IntPtr k, out UIntPtr n) =>
                        md_result_lpr_keypoints(h, new UIntPtr((uint)i), out k, out n)));
                var kps = new PointF[kpsArr.Length];
                for (int j = 0; j < kpsArr.Length; j++)
                    kps[j] = new PointF(kpsArr[j].x, kpsArr[j].y);
                list.Add(new LprResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Plate = ResultReader.ReadString(plate) ?? string.Empty,
                    Color = ResultReader.ReadString(color) ?? string.Empty,
                    Score = it.score,
                    KeyPoints = kps
                });
            }
            return list.ToArray();
        }
    }

    public sealed class PedestrianAttributeModel : BaseModel
    {
        private PedestrianAttributeModel(IntPtr handle) : base(handle) { }

        /// <summary>深拷贝模型（独立实例，可并行使用）。</summary>
        public PedestrianAttributeModel Clone() => new PedestrianAttributeModel(CloneNative());

        public PedestrianAttributeModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_PED_ATTR, modelPath, opt) { }

        public Prediction<AttributeResult> Predict(VisionImage image)
            => MakePrediction(image, ReadAttribute);

        private static AttributeResult[] ReadAttribute(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDAttrItem>(result, md_result_attribute);
            var list = new List<AttributeResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                md_result_attr_scores(result, new UIntPtr((uint)i), out var scores, out var n);
                list.Add(new AttributeResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    BoxLabelId = it.box_label_id,
                    BoxScore = it.box_score,
                    AttrScores = ResultReader.ReadFloats(scores, n)
                });
            }
            return list.ToArray();
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 子模型（OCR / LPR / insightface 组件，可独立部署）
    // ═══════════════════════════════════════════════════════════

    /// <summary>OCR 文本检测子模型（对应 C++ DBDetector）。modelPath 为 det.onnx。</summary>
    public sealed class DbDetectorModel : BaseModel
    {
        private DbDetectorModel(IntPtr handle) : base(handle) { }
        public DbDetectorModel Clone() => new DbDetectorModel(CloneNative());
        public DbDetectorModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_OCR_DET, modelPath, opt) { }

        public Prediction<OcrResult> Predict(VisionImage image)
            => MakePrediction(image, ReadOcr);

        private static OcrResult[] ReadOcr(IntPtr result)
        {
            md_result_count(result, out var count);
            var list = new List<OcrResult>((int)count);
            for (uint i = 0; i < (uint)count; i++)
            {
                var status = md_result_ocr(result, new UIntPtr(i), out var quad, out var text, out var score);
                if (status != MDStatus.MD_OK) break;
                var q = new int[8];
                Marshal.Copy(quad, q, 0, 8);
                list.Add(new OcrResult { Quad = q, Text = ResultReader.ReadString(text) ?? string.Empty, Score = score });
            }
            return list.ToArray();
        }
    }

    /// <summary>OCR 文本识别子模型（对应 C++ Recognizer）。modelPath 为 rec.onnx|dict.txt。</summary>
    public sealed class RecognizerModel : BaseModel
    {
        private RecognizerModel(IntPtr handle) : base(handle) { }
        public RecognizerModel Clone() => new RecognizerModel(CloneNative());
        public RecognizerModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_OCR_REC, modelPath, opt) { }

        public Prediction<OcrResult> Predict(VisionImage image)
            => MakePrediction(image, ReadOcr);

        private static OcrResult[] ReadOcr(IntPtr result)
        {
            md_result_count(result, out var count);
            var list = new List<OcrResult>((int)count);
            for (uint i = 0; i < (uint)count; i++)
            {
                var status = md_result_ocr(result, new UIntPtr(i), out var quad, out var text, out var score);
                if (status != MDStatus.MD_OK) break;
                var q = new int[8];
                Marshal.Copy(quad, q, 0, 8);
                list.Add(new OcrResult { Quad = q, Text = ResultReader.ReadString(text) ?? string.Empty, Score = score });
            }
            return list.ToArray();
        }
    }

    /// <summary>OCR 方向分类子模型（对应 C++ Classifier）。modelPath 为 cls.onnx。</summary>
    public sealed class OcrClassifierModel : BaseModel
    {
        private OcrClassifierModel(IntPtr handle) : base(handle) { }
        public OcrClassifierModel Clone() => new OcrClassifierModel(CloneNative());
        public OcrClassifierModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_OCR_CLS, modelPath, opt) { }

        public Prediction<OcrResult> Predict(VisionImage image)
            => MakePrediction(image, ReadOcr);

        private static OcrResult[] ReadOcr(IntPtr result)
        {
            md_result_count(result, out var count);
            var list = new List<OcrResult>((int)count);
            for (uint i = 0; i < (uint)count; i++)
            {
                var status = md_result_ocr(result, new UIntPtr(i), out var quad, out var text, out var score);
                if (status != MDStatus.MD_OK) break;
                var q = new int[8];
                Marshal.Copy(quad, q, 0, 8);
                list.Add(new OcrResult { Quad = q, Text = ResultReader.ReadString(text) ?? string.Empty, Score = score });
            }
            return list.ToArray();
        }
    }

    /// <summary>车牌检测子模型（对应 C++ LprDetection）。modelPath 为 det.onnx。</summary>
    public sealed class LprDetectionModel : BaseModel
    {
        private LprDetectionModel(IntPtr handle) : base(handle) { }
        public LprDetectionModel Clone() => new LprDetectionModel(CloneNative());
        public LprDetectionModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_LPR_DET, modelPath, opt) { }

        public Prediction<LprDetResult> Predict(VisionImage image)
            => MakePrediction(image, ReadLprDet);

        private static LprDetResult[] ReadLprDet(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDLprItem>(result, md_result_lpr);
            var list = new List<LprDetResult>(items.Length);
            foreach (var it in items)
                list.Add(new LprDetResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Score = it.score
                });
            return list.ToArray();
        }
    }

    /// <summary>车牌识别子模型（对应 C++ LprRecognizer）。modelPath 为 rec.onnx。</summary>
    public sealed class LprRecognizerModel : BaseModel
    {
        private LprRecognizerModel(IntPtr handle) : base(handle) { }
        public LprRecognizerModel Clone() => new LprRecognizerModel(CloneNative());
        public LprRecognizerModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_LPR_REC, modelPath, opt) { }

        public Prediction<LprResult> Predict(VisionImage image)
            => MakePrediction(image, ReadLpr);

        private static LprResult[] ReadLpr(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDLprItem>(result, md_result_lpr);
            var list = new List<LprResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                md_result_plate(result, new UIntPtr((uint)i), out var plate, out var color);
                list.Add(new LprResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Plate = ResultReader.ReadString(plate) ?? string.Empty,
                    Color = ResultReader.ReadString(color) ?? string.Empty,
                    Score = it.score
                });
            }
            return list.ToArray();
        }
    }

    /// <summary>insightface 人脸检测子模型（对应 C++ InsightFaceDet）。modelPath 为 det_10g.onnx。</summary>
    public sealed class InsightFaceDetModel : BaseModel
    {
        private InsightFaceDetModel(IntPtr handle) : base(handle) { }
        public InsightFaceDetModel Clone() => new InsightFaceDetModel(CloneNative());
        public InsightFaceDetModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_INSIGHTFACE_DET, modelPath, opt) { }

        public Prediction<FaceDetResult> Predict(VisionImage image)
            => MakePrediction(image, ReadFaceDet);

        private static FaceDetResult[] ReadFaceDet(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDFaceItem>(result, md_result_face);
            var list = new List<FaceDetResult>(items.Length);
            for (int i = 0; i < items.Length; i++)
            {
                var it = items[i];
                var kpsArr = ResultReader.ReadItems<MDPointF>(result,
                    new ResultReader.ItemGetter((IntPtr h, out IntPtr k, out UIntPtr n) =>
                        md_result_face_kps(h, new UIntPtr((uint)i), out k, out n)));
                var kps = new PointF[kpsArr.Length];
                for (int j = 0; j < kpsArr.Length; j++)
                    kps[j] = new PointF(kpsArr[j].x, kpsArr[j].y);
                list.Add(new FaceDetResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Score = it.score,
                    KeyPoints = kps
                });
            }
            return list.ToArray();
        }
    }

    /// <summary>人脸识别 Pipeline 子模型（检测 + 特征提取，对应 C++ FaceRecognizerPipeline）。
    /// modelPath 为 det.onnx|rec.onnx。</summary>
    public sealed class FaceRecognizerPipelineModel : BaseModel
    {
        private FaceRecognizerPipelineModel(IntPtr handle) : base(handle) { }
        public FaceRecognizerPipelineModel Clone() => new FaceRecognizerPipelineModel(CloneNative());
        public FaceRecognizerPipelineModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_REC_PIPELINE, modelPath, opt) { }

        public Prediction<FaceRecResult> Predict(VisionImage image)
            => MakePrediction(image, ReadFaceRec);

        private static FaceRecResult[] ReadFaceRec(IntPtr result)
        {
            md_result_count(result, out var count);
            var list = new List<FaceRecResult>((int)count);
            for (uint i = 0; i < (uint)count; i++)
            {
                var status = md_result_face_embedding(result, new UIntPtr(i), out var emb, out var n);
                if (status != MDStatus.MD_OK) break;
                list.Add(new FaceRecResult { Embedding = ResultReader.ReadFloats(emb, n) });
            }
            return list.ToArray();
        }
    }

    /// <summary>人脸年龄/性别子模型（对应 C++ InsightFaceGenderAge）。modelPath 为 genderage.onnx。</summary>
    public sealed class FaceGenderAgeModel : BaseModel
    {
        private FaceGenderAgeModel(IntPtr handle) : base(handle) { }
        public FaceGenderAgeModel Clone() => new FaceGenderAgeModel(CloneNative());
        public FaceGenderAgeModel(string modelPath, RuntimeOption2 opt = null)
            : base(MDModelKind.MD_MODEL_FACE_AS, modelPath, opt) { }

        public Prediction<FaceGenderAgeResult> Predict(VisionImage image)
            => MakePrediction(image, ReadGenderAge);

        private static FaceGenderAgeResult[] ReadGenderAge(IntPtr result)
        {
            var items = ResultReader.ReadItems<MDInsightFaceItem>(result, md_result_insightface);
            var list = new List<FaceGenderAgeResult>(items.Length);
            foreach (var it in items)
                list.Add(new FaceGenderAgeResult
                {
                    Box = new RectF(it.x, it.y, it.w, it.h),
                    Gender = it.gender,
                    Age = it.age
                });
            return list.ToArray();
        }
    }
}
