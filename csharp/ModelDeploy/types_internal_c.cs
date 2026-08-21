using System;
using System.Runtime.InteropServices;

//
// capi 原生类型映射（types_internal_c）
// 句柄一律用 IntPtr；结果项结构为 blittable（纯数值、无指针），供 Marshal.PtrToStructure 循环读取。
//

namespace ModelDeploy
{
    namespace types_internal_c
    {
        /* ==================== 枚举（与 md_capi.h 对齐） ==================== */

        public enum MDStatus
        {
            MD_OK = 0,
            MD_ERR_NULL_POINTER,
            MD_ERR_INVALID_ARGUMENT,
            MD_ERR_PATH_NOT_FOUND,
            MD_ERR_MODEL_LOAD,
            MD_ERR_MODEL_PREDICT,
            MD_ERR_MODEL_INIT,
            MD_ERR_UNSUPPORTED_TYPE,
            MD_ERR_UNSUPPORTED_BACKEND,
            MD_ERR_OUT_OF_MEMORY,
            MD_ERR_IMAGE_DECODE,
            MD_ERR_BUSY,
            MD_ERR_NOT_IMPLEMENTED,
            MD_ERR_AUDIO_DECODE
        }

        public enum MDModelKind
        {
            MD_MODEL_DETECTION = 0,
            MD_MODEL_CLASSIFICATION,
            MD_MODEL_POSE,
            MD_MODEL_OBB,
            MD_MODEL_INSTANCE_SEG,
            MD_MODEL_SEM_SEG,
            MD_MODEL_DEPTH,
            MD_MODEL_FACE_DET,
            MD_MODEL_FACE_REC,
            MD_MODEL_FACE_AGE,
            MD_MODEL_FACE_GENDER,
            MD_MODEL_FACE_AS,
            MD_MODEL_FACE_AS_PIPELINE,
            MD_MODEL_FACE_REC_PIPELINE,
            MD_MODEL_INSIGHTFACE,
            MD_MODEL_INSIGHTFACE_DET,
            MD_MODEL_OCR,
            MD_MODEL_OCR_DET,
            MD_MODEL_OCR_REC,
            MD_MODEL_OCR_CLS,
            MD_MODEL_LPR_DET,
            MD_MODEL_LPR_REC,
            MD_MODEL_LPR_PIPELINE,
            MD_MODEL_PED_ATTR,
            MD_MODEL_ASR,
            MD_MODEL_TTS
        }

        public enum MDResultKind
        {
            MD_RES_DETECTION = 0,
            MD_RES_CLASSIFICATION,
            MD_RES_POSE,
            MD_RES_OBB,
            MD_RES_INSTANCE_SEG,
            MD_RES_SEM_SEG,
            MD_RES_DEPTH,
            MD_RES_FACE,
            MD_RES_FACE_REC,
            MD_RES_INSIGHTFACE,
            MD_RES_OCR,
            MD_RES_LPR,
            MD_RES_ATTR,
            MD_RES_AGE,
            MD_RES_GENDER,
            MD_RES_ASR,
            MD_RES_TTS
        }

        /* ==================== 通用几何 / 颜色（blittable） ==================== */

        [StructLayout(LayoutKind.Sequential)]
        public struct MDBox
        {
            public float x, y, w, h;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPointF
        {
            public float x, y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPoint3F
        {
            public float x, y, z;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDRotatedBoxF
        {
            public float cx, cy, w, h, angle;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDColorRGBA
        {
            public byte r, g, b, a;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDLabelItem
        {
            public int id;
            public IntPtr name;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDDrawOptions
        {
            public double threshold;
            public IntPtr label_map;   // MDLabelItem*
            public UIntPtr label_map_size;
            public IntPtr font_path;   // UTF-8 string
            public int font_size;
            public double alpha;
            public int save_result;
        }

        /* ==================== 结果项结构（blittable：数组式 getter 返回其数组） ==================== */

        [StructLayout(LayoutKind.Sequential)]
        public struct MDDetectionItem
        {
            public float x, y, w, h;
            public float score;
            public int label_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDClassifyItem
        {
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPoseItem
        {
            public float x, y, w, h;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDObbItem
        {
            public float cx, cy, w, h, angle;
            public float score;
            public int label_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDIsegItem
        {
            public float x, y, w, h;
            public float score;
            public int label_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDLprItem
        {
            public float x, y, w, h;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDAttrItem
        {
            public float x, y, w, h;
            public float box_score;
            public int box_label_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceItem
        {
            public float x, y, w, h;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDInsightFaceItem
        {
            public float x, y, w, h;
            public float score;
            public int gender;
            public int age;
        }
    }
}
