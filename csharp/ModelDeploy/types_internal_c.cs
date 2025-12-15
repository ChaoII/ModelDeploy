using System;
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

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPoint
        {
            public int x;
            public int y;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDPoint3f
        {
            public float x;
            public float y;
            public float z;
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
        public struct MDRotatedRect
        {
            public float xc;
            public float yc;
            public float width;
            public float height;
            public float angle; //deg
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
            public string det_model_file;
            public string cls_model_file;
            public string rec_model_file;
            public string dict_path;
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
        public struct MDStructureTableModelParameters
        {
            public string det_model_file;
            public string rec_model_file;
            public string table_model_file;
            public string rec_label_file;
            public string table_char_dict_path;
            public int max_side_len;
            public double det_db_thresh;
            public double det_db_box_thresh;
            public double det_db_unclip_ratio;
            public string det_db_score_mode;
            public int use_dilation;
            public int rec_batch_size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDSenseVoiceParameters
        {
            public string model_path;
            public int use_itn;
            public string language;
            public string tokens_path;
            public int num_threads;
            public int debug;
        }


        [StructLayout(LayoutKind.Sequential)]
        public struct MDKokoroParameters
        {
            public string model_path;
            public string tokens_path;
            public string lexicons_en_path;
            public string lexicons_zh_path;
            public string voice_bin_path;
            public string jieba_dir;
            public string text_normalization_dir;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDOCRResult
        {
            public MDPolygon box;
            public IntPtr text;
            public float score;
            public MDPolygon table_boxes;
            public IntPtr table_structure;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDOCRResults
        {
            public IntPtr data;
            public IntPtr table_html;
            public int size;
        }


        [StructLayout(LayoutKind.Sequential)]
        public struct MDClassificationResult
        {
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDClassificationResults
        {
            public IntPtr data;
            public int size;
        }


        [StructLayout(LayoutKind.Sequential)]
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
        public struct MDMask
        {
            public IntPtr buffer;
            public int buffer_size;
            public IntPtr shape;
            public int num_dims;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDISegResult
        {
            public MDRect box;
            public MDMask mask;
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDISegResults
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDObbResult
        {
            public MDRotatedRect box;
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDObbResults
        {
            public IntPtr data;
            public int size;
        }


        [StructLayout(LayoutKind.Sequential)]
        public struct MDKeyPointResult
        {
            public MDRect box;
            public IntPtr keypoints_data;
            public int keypoints_size;
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDKeyPointResults
        {
            public IntPtr data;
            public int size;
        }
        

        [StructLayout(LayoutKind.Sequential)]
        public struct MDLPRResult
        {
            public MDRect box;
            public IntPtr landmarks_data;
            public int landmarks_size;
            public int label_id;
            public float score;
            public IntPtr car_plate_str;
            public IntPtr car_plate_color;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDLPRResults
        {
            public IntPtr data;
            public int size;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        public struct MDAttributeResult
        {
            public MDRect box;
            public int box_label_id;
            public float box_score;
            public IntPtr attr_scores;
            public int attr_scores_size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDAttributeResults
        {
            public IntPtr data;
            public int size;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceRecognizerResult
        {
            public IntPtr embedding;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceRecognizerResults
        {
            public IntPtr data;
            public int size;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceAsSecondResult
        {
            public int label_id;
            public float score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceAsSecondResults
        {
            public IntPtr data;
            public int size;
        }

        public enum MDFaceAsResult
        {
            Real = 0,
            Fuzzy = 1,
            Spoof = 2
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDFaceAsResults
        {
            public IntPtr data;
            public int size;
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

        [StructLayout(LayoutKind.Sequential)]
        public struct MDTTSResult
        {
            public IntPtr data;
            public int size;
            public int sample_rate;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDRuntimeOption
        {
            public string trt_min_shape;
            public string trt_opt_shape;
            public string trt_max_shape;
            public string trt_engine_cache_path;
            public int enable_fp16;
            public int cpu_thread_num;
            public int device_id;
            public int enable_trt;
            public Device device;
            public Backend backend;
            public int graph_opt_level;
            public string password;
        }
    }
}