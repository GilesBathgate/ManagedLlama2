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

        // Full generation from scratch
        var transformer1 = new Transformer(modelPath, tokenizerPath);
        var tokensFull = transformer1.Generate("You are a helpful assistant.", 20);
        var fullSb = new StringBuilder();
        foreach (var t in tokensFull) fullSb.Append(t);

        // Generation with rollback: generate 10 tokens, rollback to position 5, then continue generation
        var transformer2 = new Transformer(modelPath, tokenizerPath);
        var tokensPart1 = transformer2.Generate("You are a helpful assistant.", 10);
        foreach (var _ in tokensPart1) { } // evaluate first 10 steps

        transformer2.Rollback(5); // rewind sequence position to N=5
        var tokensResumed = transformer2.Generate("You are a helpful assistant.", 20);
        var resumedSb = new StringBuilder();
        foreach (var t in tokensResumed) resumedSb.Append(t);

        Assert.Equal(fullSb.ToString(), resumedSb.ToString());
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
