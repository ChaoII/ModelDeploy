using ModelDeploy;
using ModelDeploy.Models;
using NUnit.Framework;

namespace ModelDeployUnitTest;

public class CapiAudioTests
{
    private static string GetTestDataDir()
    {
        return Environment.GetEnvironmentVariable("TEST_DATA_DIR")
            ?? Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..");
    }

    private static string ModelRoot => Path.Combine(GetTestDataDir(), "test_data", "test_models", "onnx");
    private static bool Has(string p) => File.Exists(p);

    private static RuntimeOption CpuOrt()
    {
        return new RuntimeOption().UseOrt().SetDevice(Device.CPU).SetCpuThreads(4);
    }

    [Test]
    public void Asr_PredictWav_ReturnsText()
    {
        var dir = Path.Combine(ModelRoot, "sense_voice");
        var model = string.Join("|",
            Path.Combine(dir, "model.int8.onnx"),
            Path.Combine(dir, "tokens.txt"));
        var wav = Path.Combine(dir, "test_wavs", "zh.wav");
        if (!Has(Path.Combine(dir, "model.int8.onnx")) || !Has(wav)) Assert.Ignore("model/wav not found");

        using var asr = new SenseVoiceModel(model, CpuOrt());
        var result = asr.PredictWav(wav);
        Assert.That(result.Text, Is.Not.Empty);
    }

    [Test]
    public void Asr_PredictWavStructured_ReturnsLabels()
    {
        var dir = Path.Combine(ModelRoot, "sense_voice");
        var model = string.Join("|",
            Path.Combine(dir, "model.int8.onnx"),
            Path.Combine(dir, "tokens.txt"));
        var wav = Path.Combine(dir, "test_wavs", "zh.wav");
        if (!Has(Path.Combine(dir, "model.int8.onnx")) || !Has(wav)) Assert.Ignore("model/wav not found");

        using var asr = new SenseVoiceModel(model, CpuOrt());
        var result = asr.PredictWavStructured(wav);
        Assert.That(result.Text, Is.Not.Empty);
        Assert.That(result.Language, Is.EqualTo("zh"));
        Assert.That(result.Event, Is.EqualTo("Speech"));
    }

    [Test]
    public void Kokoro_Create_LoadsModel()
    {
        // 注意：Kokoro predict 需 s2t_map.bin/t2s_map.bin（测试数据未携带），此处仅验证模型可加载。
        var dir = Path.Combine(ModelRoot, "kokoro_v1_1");
        var model = string.Join("|",
            Path.Combine(dir, "model.onnx"),
            Path.Combine(dir, "tokens.txt"),
            Path.Combine(dir, "lexicon-gb-en.txt"),
            Path.Combine(dir, "lexicon-zh.txt"),
            Path.Combine(dir, "voices.bin"),
            Path.Combine(dir, "dict"),
            dir);
        if (!Has(Path.Combine(dir, "model.onnx"))) Assert.Ignore("model not found");

        using var tts = new KokoroModel(model, CpuOrt());
        Assert.That(tts.IsReady(), Is.True);
    }
}
