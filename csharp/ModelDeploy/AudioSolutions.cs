using System;
using System.Runtime.InteropServices;
using ModelDeploy.types_internal_c;
using static ModelDeploy.NativeMethods;

namespace ModelDeploy.Audio
{
    /// <summary>说话人检索（对应 CAPI md_audio_speaker_search_*）。</summary>
    public sealed class SpeakerSearch : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public SpeakerSearch()
        {
            if (md_audio_solution_create(out _handle, MDAudioSolutionKind.MD_AUDIO_SPEAKER_SEARCH) != MDStatus.MD_OK)
                throw new InvalidOperationException("audio solution create failed");
        }

        public void Enroll(string label, float[] embedding)
        {
            var lp = Utf8.Alloc(label);
            try
            {
                if (md_audio_speaker_search_enroll(_handle, lp, embedding, new UIntPtr((uint)embedding.Length)) != MDStatus.MD_OK)
                    throw new InvalidOperationException("enroll failed");
            }
            finally { Utf8.Free(lp); }
        }

        public string Match(float[] embedding)
        {
            var status = md_audio_speaker_search_match(_handle, embedding, new UIntPtr((uint)embedding.Length), 1,
                out var label, out _);
            if (status != MDStatus.MD_OK) throw new InvalidOperationException("match failed");
            return ResultReader.ReadString(label) ?? string.Empty;
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero && !_disposed)
            {
                md_audio_solution_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }
    }

    /// <summary>音频工具（对应 CAPI md_audio_resample）。</summary>
    public static class Tools
    {
        public static float[] Resample(float[] input, int inSr, int outSr)
        {
            if (md_audio_resample(input, new UIntPtr((uint)input.Length), inSr, outSr, out var ptr, out var n) != MDStatus.MD_OK)
                throw new InvalidOperationException("resample failed");
            var count = (int)n;
            var result = new float[count];
            Marshal.Copy(ptr, result, 0, count);
            return result;
        }
    }
}
