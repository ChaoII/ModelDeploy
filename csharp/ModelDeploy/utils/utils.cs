using System;
using System.Runtime.InteropServices;
using System.Text;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.utils
{
    public static class Utils
    {
        /// <summary>
        /// 将托管字符串转换为 UTF-8 非托管内存，调用者需负责释放。
        /// </summary>
        public static IntPtr ConvertStringToHGlobalUtf8(string str)
        {
            if (string.IsNullOrEmpty(str))
            {
                IntPtr ptr = Marshal.AllocHGlobal(1);
                Marshal.WriteByte(ptr, 0);
                return ptr;
            }

            byte[] utf8Bytes = Encoding.UTF8.GetBytes(str);
            IntPtr unmanagedPtr = Marshal.AllocHGlobal(utf8Bytes.Length + 1);
            Marshal.Copy(utf8Bytes, 0, unmanagedPtr, utf8Bytes.Length);
            Marshal.WriteByte(unmanagedPtr + utf8Bytes.Length, 0); // null-terminated
            return unmanagedPtr;
        }

        /// <summary>
        /// 从 UTF-8 非托管指针转换为托管字符串。
        /// </summary>
        public static string PtrToStringUTF8(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return string.Empty;

            int len = 0;
            while (Marshal.ReadByte(ptr, len) != 0)
                len++;

            byte[] buffer = new byte[len];
            Marshal.Copy(ptr, buffer, 0, len);
            return Encoding.UTF8.GetString(buffer);
        }

        /// <summary>
        /// 检查返回码并抛出异常。
        /// </summary>
        public static void Check(int ret, string context)
        {
            if (ret != 0)
                throw new InvalidOperationException($"{context} failed (error code: {ret})");
        }

        /// <summary>
        /// 判断图像中按钮是否可用。
        /// </summary>
        public static bool GetButtonEnableStatus(Image image, int pixThreshold, double rateThreshold)
        {
            return md_get_button_enable_status(ref image.RawImage, pixThreshold, rateThreshold);
        }

        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool md_get_button_enable_status(ref MDImage image, int pixThreshold,
            double rateThreshold);
    }

    // 自动释放的 UTF8 字符串封装
    internal class Utf8String : IDisposable
    {
        public IntPtr Ptr { get; }

        public Utf8String(string str)
        {
            Ptr = Utils.ConvertStringToHGlobalUtf8(str);
        }

        public void Dispose()
        {
            if (Ptr != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Ptr);
                GC.SuppressFinalize(this);
            }
        }
    }
}