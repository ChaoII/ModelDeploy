using System.Runtime.InteropServices;
using ModelDeploy.utils;
using ModelDeploy.types_internal_c;
using System.Collections.Generic;
using System;

namespace ModelDeploy.audio.tts
{
    public class TtsResult
    {
        private List<float> Data { get; }
        private int SampleRate { get; }

        private TtsResult(List<float> data, int sampleRate)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            SampleRate = sampleRate;
        }

        public static TtsResult FromNative(MDTTSResult cResult)
        {
            float[] buffer = new float[cResult.size];
            Marshal.Copy(cResult.data, buffer, 0, cResult.size);
            return new TtsResult(new List<float>(buffer), cResult.sample_rate);
        }

        public MDTTSResult ToNative()
        {
            var cResult = new MDTTSResult
            {
                size = Data.Count,
                sample_rate = SampleRate,
                data = Marshal.AllocHGlobal(Data.Count * sizeof(float))
            };

            float[] buffer = Data.ToArray();
            Marshal.Copy(buffer, 0, cResult.data, buffer.Length);
            return cResult;
        }
    }
}