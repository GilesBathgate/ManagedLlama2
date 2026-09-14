using System;
using System.Text;
using Xunit;

namespace libLlama2.UnitTests;

public class RollbackTests
{
    [Fact(Skip = "Requires CUDA GPU device")]
    public void Test_RunState_Rollback_ValidPositions()
    {
        using var cudaContext = new ManagedCuda.CudaContext(0);
        var config = new Config { seqLength = 100, dim = 128, numLayers = 1, vocabSize = 100, numHeads = 4, numKVHeads = 4, hiddenDim = 256 };
        var runstate = new RunState(cudaContext, ref config, 128);

        runstate.Position = 10;
        Assert.Equal(10, runstate.Position);

        runstate.Rollback(5);
        Assert.Equal(5, runstate.Position);

        runstate.Rollback(0);
        Assert.Equal(0, runstate.Position);
    }

    [Fact(Skip = "Requires CUDA GPU device")]
    public void Test_RunState_Rollback_InvalidPositions_ThrowsException()
    {
        using var cudaContext = new ManagedCuda.CudaContext(0);
        var config = new Config { seqLength = 100, dim = 128, numLayers = 1, vocabSize = 100, numHeads = 4, numKVHeads = 4, hiddenDim = 256 };
        var runstate = new RunState(cudaContext, ref config, 128);

        runstate.Position = 10;

        Assert.Throws<ArgumentOutOfRangeException>(() => runstate.Rollback(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => runstate.Rollback(11));
    }

    [Fact(Skip = "Requires CUDA GPU device and model file")]
    public void Test_Chat_Baseline()
    {
        var modelPath = "model-7b.bin";
        var tokenizerPath = "tokenizer.bin";
        var systemPrompt = "You are a helpful assistant.";

        var transformer = new Transformer(modelPath, tokenizerPath, temperature: 0.0f);

        // Turn 1
        var tokensTurn1 = transformer.Chat(systemPrompt, new[] { "Hello" });
        var sb1 = new StringBuilder();
        foreach (var t in tokensTurn1) sb1.Append(t);
        var output1 = sb1.ToString();

        Assert.False(string.IsNullOrWhiteSpace(output1));

        // Turn 2
        var tokensTurn2 = transformer.Chat(systemPrompt, new[] { "What is 2+2?" });
        var sb2 = new StringBuilder();
        foreach (var t in tokensTurn2) sb2.Append(t);

        var output2 = sb2.ToString();
        Assert.False(string.IsNullOrWhiteSpace(output2));
    }

    [Fact(Skip = "Requires CUDA GPU device and model file")]
    public void Test_Chat_MultiTurn_Control()
    {
        var modelPath = "model-7b.bin";
        var tokenizerPath = "tokenizer.bin";
        var systemPrompt = "You are a helpful assistant.";

        var transformer = new Transformer(modelPath, tokenizerPath, temperature: 0.0f);

        // Turn 1
        var tokensTurn1 = transformer.Chat(systemPrompt, new[] { "Hello" });
        foreach (var _ in tokensTurn1) { }

        // Turn 2
        var tokensTurn2 = transformer.Chat(systemPrompt, new[] { "What is 2+2?" });
        var sb = new StringBuilder();
        foreach (var t in tokensTurn2) sb.Append(t);

        var output = sb.ToString();
        Assert.False(string.IsNullOrWhiteSpace(output));
    }

    [Fact(Skip = "Requires CUDA GPU device and model file")]
    public void Test_Chat_ThreeTurn_Control()
    {
        var modelPath = "model-7b.bin";
        var tokenizerPath = "tokenizer.bin";
        var systemPrompt = "You are a helpful assistant.";

        var transformer = new Transformer(modelPath, tokenizerPath, temperature: 0.0f);

        // Turn 1
        var tokensTurn1 = transformer.Chat(systemPrompt, new[] { "Hello" });
        foreach (var _ in tokensTurn1) { }

        // Turn 2
        var tokensTurn2 = transformer.Chat(systemPrompt, new[] { "What is 2+2?" });
        foreach (var _ in tokensTurn2) { }

        // Turn 3
        var tokensTurn3 = transformer.Chat(systemPrompt, new[] { "What is 2+4?" });
        var sb = new StringBuilder();
        foreach (var t in tokensTurn3) sb.Append(t);

        var output = sb.ToString();
        Assert.False(string.IsNullOrWhiteSpace(output));
    }

    [Fact(Skip = "Requires CUDA GPU device and model file")]
    public void Test_Chat_MultiTurn_Rollback()
    {
        var modelPath = "model-7b.bin";
        var tokenizerPath = "tokenizer.bin";
        var systemPrompt = "You are a helpful assistant.";

        // Control run: Turn 1 ("Hello"), then Turn 2 ("What is 2+4?")
        var transformerControl = new Transformer(modelPath, tokenizerPath, temperature: 0.0f);
        var controlTurn1 = transformerControl.Chat(systemPrompt, new[] { "Hello" });
        foreach (var _ in controlTurn1) { }
        var controlTurn2 = transformerControl.Chat(systemPrompt, new[] { "What is 2+4?" });
        var sbControl = new StringBuilder();
        foreach (var t in controlTurn2) sbControl.Append(t);

        // Branching run: Turn 1 ("Hello"), Turn 2 ("What is 2+2?"), rollback to posAfterTurn1, then Turn 2 ("What is 2+4?")
        var transformerRollback = new Transformer(modelPath, tokenizerPath, temperature: 0.0f);
        var rbTurn1 = transformerRollback.Chat(systemPrompt, new[] { "Hello" });
        foreach (var _ in rbTurn1) { }
        var posAfterTurn1 = transformerRollback.Position;

        var rbTurn2A = transformerRollback.Chat(systemPrompt, new[] { "What is 2+2?" });
        foreach (var _ in rbTurn2A) { }

        transformerRollback.Rollback(posAfterTurn1);
        var rbTurn2B = transformerRollback.Chat(systemPrompt, new[] { "What is 2+4?" });
        var sbRollback = new StringBuilder();
        foreach (var t in rbTurn2B) sbRollback.Append(t);

        Assert.False(string.IsNullOrWhiteSpace(sbRollback.ToString()));
        Assert.Equal(sbControl.ToString(), sbRollback.ToString());
    }

    [Fact]
    public void Test_PositionTracker_Logic()
    {
        var tracker = new PositionTracker(10);
        Assert.Equal(10, tracker.Position);

        tracker.Rollback(5);
        Assert.Equal(5, tracker.Position);

        tracker.Rollback(0);
        Assert.Equal(0, tracker.Position);
    }

    private class PositionTracker
    {
        public int Position { get; set; }

        public PositionTracker(int initialPosition)
        {
            Position = initialPosition;
        }

        public void Rollback(int position)
        {
            if (position < 0 || position > Position)
            {
                throw new ArgumentOutOfRangeException(nameof(position), $"Position must be between 0 and current position ({Position}).");
            }
            Position = position;
        }
    }
}
