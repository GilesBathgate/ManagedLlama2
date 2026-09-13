using System;
using System.Text;
using Xunit;

namespace libLlama2.UnitTests;

public class RollbackTests
{
    [Fact(Skip = "Requires CUDA GPU device and model file")]
    public void Test_Transformer_Rollback_Equivalence()
    {
        var modelPath = "model-7b.bin";
        var tokenizerPath = "tokenizer.bin";
        var systemPrompt = "You are a helpful assistant.";

        var transformer = new Transformer(modelPath, tokenizerPath);

        // Turn 1
        var tokensTurn1 = transformer.Chat(systemPrompt, new[] { "Hello" });
        foreach (var _ in tokensTurn1) { }
        var posAfterTurn1 = transformer.Position;

        // Turn 2
        var tokensTurn2A = transformer.Chat(systemPrompt, new[] { "What is 2+2?" });
        var sb2A = new StringBuilder();
        foreach (var t in tokensTurn2A) sb2A.Append(t);

        // Rollback back to Turn 1 position and repeat Turn 2
        transformer.Rollback(posAfterTurn1);
        var tokensTurn2B = transformer.Chat(systemPrompt, new[] { "What is 2+2?" });
        var sb2B = new StringBuilder();
        foreach (var t in tokensTurn2B) sb2B.Append(t);

        Assert.Equal(sb2A.ToString(), sb2B.ToString());
    }

    [Fact]
    public void Test_PositionTracking_Rollback_ValidPositions()
    {
        var state = new SequenceState();
        state.Position = 10;
        Assert.Equal(10, state.Position);

        state.Rollback(5);
        Assert.Equal(5, state.Position);

        state.Rollback(0);
        Assert.Equal(0, state.Position);
    }

    [Fact]
    public void Test_PositionTracking_Rollback_InvalidPositions_ThrowsException()
    {
        var state = new SequenceState();
        state.Position = 10;

        Assert.Throws<ArgumentOutOfRangeException>(() => state.Rollback(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => state.Rollback(11));
    }

    private class SequenceState
    {
        public int Position { get; set; }

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
