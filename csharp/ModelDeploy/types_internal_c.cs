using System.Runtime.InteropServices;


namespace ModelDeploy
{
    namespace types_internal_c
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct MDModel
        {
            string model_name;
            MDModelType type;
            MDModelFormat format;
            IntPtr model_content;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPoint
        {
            public int x;
            public int y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPointF
        {
            public double x;
            public double y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDRect
        {
            public int x;
            public int y;
            public int width;
            public int height;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDImage
        {
            public int width;
            public int height;
            public int channels;
            public IntPtr data;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPolygon
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDOCRModelParameters
        {
            /// ocr model directory;
            public string model_dir;

            /// ocr dictionary path
            public string dict_path;

            /// thread num default is 8
            public int thread_num;

            /// model format default is PaddlePaddle
            public MDModelFormat format;

            /// maximum side length default 960
            public int max_side_len;

            /// db threshold default 0.3
            public double det_db_thresh;

            /// db box threshold default 0.6
            public double det_db_box_thresh;

            /// detect db unclip ratio default 1.5
            public double det_db_unclip_ratio;

            /// detect db score mode default is "slow"
            public string det_db_score_mode;

            /// is use dilation default is false(0)
            public int use_dilation;

            /// recognition batch size default is 8, unusually set the same as thread_num
            public int rec_batch_size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDOCRResult
        {
            public MDPolygon box;
            public IntPtr text;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDOCRResults
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDColor
        {
            public byte r;
            public byte g;
            public byte b;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDSize
        {
            public int width;
            public int height;
        }

        public struct MDDetectionResult
        {
            public MDRect box;
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDDetectionResults
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDLandMarkResult
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceFeature
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDEyeStateResult
        {
            public MDEyeState left_eye;
            public MDEyeState right_eye;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDASRResult
        {
            public IntPtr msg;
            public IntPtr stamp;
            public IntPtr stamp_sents;
            public IntPtr tpass_msg;
            public float snippet_time;
        }
    }
}