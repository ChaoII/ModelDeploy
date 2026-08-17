using System;

namespace ModelDeploy.V2
{
    /// <summary>浮点矩形。</summary>
    public readonly struct RectF
    {
        public float X { get; }
        public float Y { get; }
        public float Width { get; }
        public float Height { get; }

        public RectF(float x, float y, float w, float h)
        {
            X = x; Y = y; Width = w; Height = h;
        }

        public override string ToString() => $"RectF({X},{Y},{Width}x{Height})";
    }

    /// <summary>旋转矩形（OBB）。</summary>
    public readonly struct RotatedRectF
    {
        public float Cx { get; }
        public float Cy { get; }
        public float Width { get; }
        public float Height { get; }
        public float Angle { get; }

        public RotatedRectF(float cx, float cy, float w, float h, float angle)
        {
            Cx = cx; Cy = cy; Width = w; Height = h; Angle = angle;
        }

        public override string ToString() => $"RotatedRectF({Cx},{Cy},{Width}x{Height}@{Angle})";
    }

    public readonly struct PointF
    {
        public float X { get; }
        public float Y { get; }
        public PointF(float x, float y) { X = x; Y = y; }
        public override string ToString() => $"({X},{Y})";
    }

    public readonly struct Point3F
    {
        public float X { get; }
        public float Y { get; }
        public float Z { get; }
        public Point3F(float x, float y, float z) { X = x; Y = y; Z = z; }
        public override string ToString() => $"({X},{Y},{Z})";
    }
}

namespace ModelDeploy.V2.Results
{
    public sealed class DetectionResult
    {
        public RectF Box { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }
        public override string ToString() => $"Det({LabelId}, {Score:F3}, {Box})";
    }

    public sealed class ClassificationResult
    {
        public int LabelId { get; set; }
        public float Score { get; set; }
        public override string ToString() => $"Cls({LabelId}, {Score:F3})";
    }

    public sealed class PoseResult
    {
        public RectF Box { get; set; }
        public float Score { get; set; }
        public Point3F[] KeyPoints { get; set; } = Array.Empty<Point3F>();
        public override string ToString() => $"Pose({Score:F3}, kps={KeyPoints.Length})";
    }

    public sealed class ObbResult
    {
        public RotatedRectF Box { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }
        public override string ToString() => $"Obb({LabelId}, {Score:F3}, {Box})";
    }

    public sealed class InstanceSegResult
    {
        public RectF Box { get; set; }
        public int LabelId { get; set; }
        public float Score { get; set; }
        public byte[] Mask { get; set; } = Array.Empty<byte>();
        public int MaskHeight { get; set; }
        public int MaskWidth { get; set; }
        public override string ToString() => $"Iseg({LabelId}, {Score:F3}, mask={MaskWidth}x{MaskHeight})";
    }

    public sealed class SemSegResult
    {
        public byte[] Labels { get; set; } = Array.Empty<byte>();
        public int Height { get; set; }
        public int Width { get; set; }
        public int NumClasses { get; set; }
    }

    public sealed class DepthResult
    {
        public float[] Depth { get; set; } = Array.Empty<float>();
        public int Height { get; set; }
        public int Width { get; set; }
    }

    public sealed class FaceDetResult
    {
        public RectF Box { get; set; }
        public float Score { get; set; }
        public PointF[] KeyPoints { get; set; } = Array.Empty<PointF>();
        public override string ToString() => $"FaceDet({Score:F3}, {Box})";
    }

    public sealed class FaceRecResult
    {
        public float[] Embedding { get; set; } = Array.Empty<float>();
    }

    public sealed class InsightFaceResult
    {
        public RectF Box { get; set; }
        public float Score { get; set; }
        public PointF[] KeyPoints { get; set; } = Array.Empty<PointF>();
        public float[] Embedding { get; set; } = Array.Empty<float>();
        public float[] Pose { get; set; } = Array.Empty<float>();
        public int Gender { get; set; } = -1;
        public int Age { get; set; } = -1;
        public override string ToString() => $"InsightFace({Score:F3}, gender={Gender}, age={Age})";
    }

    public sealed class OcrResult
    {
        public int[] Quad { get; set; } = Array.Empty<int>();
        public string Text { get; set; } = string.Empty;
        public float Score { get; set; }
        public int ClsLabel { get; set; }
        public float ClsScore { get; set; }
        public override string ToString() => $"OCR('{Text}', {Score:F3})";
    }

    public sealed class LprResult
    {
        public RectF Box { get; set; }
        public string Plate { get; set; } = string.Empty;
        public string Color { get; set; } = string.Empty;
        public float Score { get; set; }
        public PointF[] KeyPoints { get; set; } = Array.Empty<PointF>();
        public override string ToString() => $"LPR('{Plate}', '{Color}', {Score:F3})";
    }

    public sealed class AttributeResult
    {
        public RectF Box { get; set; }
        public int BoxLabelId { get; set; }
        public float BoxScore { get; set; }
        public float[] AttrScores { get; set; } = Array.Empty<float>();
        public override string ToString() => $"Attr({BoxLabelId}, {BoxScore:F3})";
    }

    public sealed class AsrResult
    {
        public string Text { get; set; } = string.Empty;
        public override string ToString() => $"ASR('{Text}')";
    }

    public sealed class TtsResult
    {
        public float[] Audio { get; set; } = Array.Empty<float>();
        public int SampleRate { get; set; }
    }

    /// <summary>人脸年龄/性别（FACE_AS 子模型结果）。</summary>
    public sealed class FaceGenderAgeResult
    {
        public RectF Box { get; set; }
        public int Gender { get; set; } = -1;
        public int Age { get; set; } = -1;
        public override string ToString() => $"FaceGenderAge(gender={Gender}, age={Age})";
    }

    /// <summary>车牌检测结果（LPR_DET 子模型，仅框 + 置信度）。</summary>
    public sealed class LprDetResult
    {
        public RectF Box { get; set; }
        public float Score { get; set; }
        public override string ToString() => $"LprDet({Score:F3}, {Box})";
    }
}
