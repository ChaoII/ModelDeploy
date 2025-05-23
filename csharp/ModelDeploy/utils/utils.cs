using System;
using System.Runtime.InteropServices;
using System.Text;
using ModelDeploy.types_internal_c;

namespace ModelDeploy.utils
{
    public class Utils
    {
        public static IntPtr ConvertStringToHGlobalUtf8(string str)
        {
            byte[] utf8Bytes = Encoding.UTF8.GetBytes(str);
            IntPtr unmanagedString = Marshal.AllocHGlobal(utf8Bytes.Length + 1);
            Marshal.Copy(utf8Bytes, 0, unmanagedString, utf8Bytes.Length);
            Marshal.WriteByte(unmanagedString + utf8Bytes.Length, 0);
            return unmanagedString;
        }
        

        public static string PtrToStringUTF8(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return null;
            }

            int length = 0;
            while (Marshal.ReadByte(ptr, length) != 0)
            {
                length++;
            }

            byte[] buffer = new byte[length];
            Marshal.Copy(ptr, buffer, 0, length);
            return Encoding.UTF8.GetString(buffer);
        }

        public bool GetButtonEnableStatus(Image image, int pixThreshold, double rateThreshold)
        {
            MDImage cImage = image.RawImage;
            bool result = md_get_button_enable_status(ref cImage, pixThreshold, rateThreshold);
            return result;
        }


        [DllImport("ModelDeploySDK.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool md_get_button_enable_status(ref MDImage image, int pixThreshold,
            double rateThreshold);
    }
}