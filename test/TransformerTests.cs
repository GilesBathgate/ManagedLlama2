namespace libLlama2.UnitTests;

public class TransformerTests
{

    [Fact(Skip = "Requires much memory")]
    public void Test_Transformer()
    {
        var prompt = new[] { 1, 887, 526, 263, 8444, 20255 };

        var transformer = new Transformer("model-7b.bin");
        transformer.Run(prompt);
    }
}