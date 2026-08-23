using ModelDeploy;
using ModelDeploy.Audio;

namespace ModelDeployUnitTest;

public class AudioNlpSolutionTests
{
    [Test]
    public void SpeakerSearch_EnrollMatch_ReturnsLabel()
    {
        using var search = new SpeakerSearch();
        search.Enroll("alice", new float[] { 1.0f, 0.0f, 0.0f });
        Assert.That(search.Match(new float[] { 0.99f, 0.1f, 0.0f }), Is.EqualTo("alice"));
    }

    [Test]
    public void AudioTools_Resample_UpsamplesByRatio()
    {
        var input = new float[800];
        var outS = Tools.Resample(input, 8000, 16000);
        Assert.That(outS.Length, Is.EqualTo(1600));
    }

    [Test]
    public void NlpTools_SplitSentences_Counts()
    {
        Assert.That(NlpTools.SplitSentences("你好。世界！"), Has.Length.EqualTo(2));
    }

    [Test]
    public void NlpTools_Stats_Words()
    {
        Assert.That(NlpTools.Stats("a b c").Words, Is.EqualTo(3));
    }
}
