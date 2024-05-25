using System.Text;

namespace libLlama2.UnitTests;

public class TransformerTests
{

    [Fact(Skip = "Requires much memory")]
    public void Test_Transformer()
    {
        var expected = "You are a helpful assistant.  You are able to assist the user in a variety of ways";

        var transformer = new Transformer("model-7b.bin");
        var tokens = transformer.Generate("You are a helpful assistant. ", 19);

        var builder = new StringBuilder();
        foreach (var token in tokens)
            builder.Append(token);

        var actual = builder.ToString();

        Assert.Equal(expected, actual);
    }

    [Theory(Skip = "Requires tokenizer.bin")]
    [InlineData("You are a helpful assistant", new[] { 1, 887, 526, 263, 8444, 20255, 2 })]
    [InlineData("<<SYS>>", new[] { 1, 3532, 14816, 29903, 6778, 2 })]
    [InlineData("[/INST]", new[] { 1, 518, 29914, 25580, 29962, 2 })]
    [InlineData("\n", new[] { 1, 29871, 13, 2 })]
    public void Test_Tokenizer(string value, int[] expected)
    {
        var tokenizer = new Tokenizer("tokenizer.bin", 32000);

        var actual = tokenizer.Encode(value, true, true);

        Assert.Equal(expected, actual);

    }
}