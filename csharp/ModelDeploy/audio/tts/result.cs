using System.Runtime.InteropServices;
using ModelDeploy.utils;
using ModelDeploy.types_internal_c;
using System.Collections.Generic;
using System;

namespace ModelDeploy.audio.tts
{
    public class TTSResult
    {
        public List<float> Data;
        public int SampleRate;

        public static TTSResult FromMDTTSResult(MDTTSResult cResult)
        {
            TTSResult result = new TTSResult();
            result.Data = new List<float>();
            for (int i = 0; i < cResult.size; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResult.data, i * Marshal.SizeOf<float>());
                float res = Marshal.PtrToStructure<float>(currentPtr);
                result.Data.Add(res);
            }
            result.SampleRate = cResult.sample_rate;
            return result;
        }

        public static MDTTSResult ToMDTTSResult(TTSResult result)
        {
            MDTTSResult cResult = new MDTTSResult();
            // notice that the data will destroy in C
            cResult.data = Marshal.AllocHGlobal(result.Data.Count * Marshal.SizeOf<float>());
            cResult.size = result.Data.Count;
            cResult.sample_rate = result.SampleRate;

            for (int i = 0; i < result.Data.Count; i++)
            {
                IntPtr currentPtr = IntPtr.Add(cResult.data, i * Marshal.SizeOf<float>());
                float res = result.Data[i];
                Marshal.StructureToPtr(res, currentPtr, false);
            }

            return cResult;
        }
    }
}