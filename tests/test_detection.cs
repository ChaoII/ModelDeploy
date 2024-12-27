using System.Runtime.InteropServices;

namespace ConsoleApp1
{
    internal class test_detection
    {

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
            public IntPtr data; // 使用IntPtr来表示指针

            public void show()
            {
                md_show_image(ref this);
            }

            public void release()
            {
                md_free_image(ref this);
            }
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
            public IntPtr data; // 使用IntPtr来表示指针
            public IntPtr size; // 使用IntPtr来表示指针

            public List<MDDetectionResult> toMDDetectionResultList()
            {
                List<MDDetectionResult> results = new List<MDDetectionResult>();
                for (int i = 0; i < (int)size; i++)
                {
                    IntPtr currentPtr = IntPtr.Add(data, i * Marshal.SizeOf<MDDetectionResult>());
                    MDDetectionResult res = Marshal.PtrToStructure<MDDetectionResult>(currentPtr);
                    results.Add(res);
                }
                return results;
            }
            public void display()
            {
                md_print_detection_result(ref this);
            }

            public void release()
            {
                md_free_detection_result(ref this);
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MDModel
        {
            public IntPtr model_name; // 使用IntPtr来表示指针
            public int type;
            public int format;
            public IntPtr model_content; // 使用IntPtr来表示指针

            public string getModelName()
            {
                string modelName = Marshal.PtrToStringUTF8(model_name);
                return modelName;
            }
            public void release()
            {
                md_free_detection_model(ref this);
            }

        }

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int md_create_detection_model(ref MDModel model, string model_dir, int thread_num = 8, int format = 0);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int md_set_detection_input_size(ref MDModel model, MDSize size);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int md_detection_predict(ref MDModel model, ref MDImage image, ref MDDetectionResults results);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_print_detection_result(ref MDDetectionResults result);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_draw_detection_result(ref MDImage image, ref MDDetectionResults result, string font_path, int font_size, double alpha, int save_result);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_free_detection_result(ref MDDetectionResults result);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_free_detection_model(ref MDModel model);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern MDImage md_read_image(string image_path);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_show_image(ref MDImage image);

        [DllImport("model_deploy_sdk.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void md_free_image(ref MDImage img);



        static void Main(string[] args)
        {
            MDModel model = new MDModel();
            int result = md_create_detection_model(ref model, "best.onnx", 8, 0);
            if (result != 0)
            {
                Console.WriteLine("Failed to create detection model.");
                return;
            }

            // 设置检测输入大小
            MDSize size = new MDSize();
            size.height = 1440;
            size.width = 1440;
            result = md_set_detection_input_size(ref model, size);
            if (result != 0)
            {
                Console.WriteLine("Failed to set detection input size.");
                md_free_detection_model(ref model);
                return;
            }

            // 读取图像
            MDImage image = md_read_image("test_detection.png");

            // 弹窗显示图像
            //image.show();

            // 执行预测
            MDDetectionResults results = new MDDetectionResults();
            result = md_detection_predict(ref model, ref image, ref results);
            if (result != 0)
            {
                Console.WriteLine("Detection predict failed.");
                image.release();
                model.release();
                return;
            }
            Console.WriteLine(model.getModelName());
            results.toMDDetectionResultList();
            // 打印检测结果
            results.display();

            // 绘制检测结果
            md_draw_detection_result(ref image, ref results, "", 12, 0.5, 1);

            // 显示图像
            image.show();

            // 释放资源
            results.release();
            image.release();
            model.release();
        }
    }
}