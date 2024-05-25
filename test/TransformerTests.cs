namespace libLlama2.UnitTests;

public class TransformerTests
{
    private static Half ToHalf(uint i) => BitConverter.ToHalf(BitConverter.GetBytes(i), 0);

    [Fact(Skip = "Requires much memory")]
    public void Test_Transformer()
    {
        var expected = new uint[] {
            0x3009, 0xb790, 0x33a1, 0xba6d, 0x3e23, 0x4181, 0x3e74, 0x4120,
            0x3d3f, 0x366a, 0x3ebd, 0x3be8, 0x38b5, 0x36fa, 0x3d1d, 0x3747 };

        var expectedLogits = expected.Select(ToHalf);

        //                 <BOS> You are a helpful assistant
        var prompt = new[] { 1, 887, 526, 263, 8444, 20255 };

        var transformer = new Transformer("model-7b.bin");
        var logits = transformer.Run(prompt);

        var actual = logits.Take(expected.Length);

        Assert.Equal(expectedLogits, actual);
    }
}