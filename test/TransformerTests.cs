using System.Text;

namespace libLlama2.UnitTests;

public class TransformerTests
{

    [Fact(Skip = "Requires much memory")]
    public void Test_Transformer()
    {
        var expected = "You are a helpful assistant.  You are able to assist the user in a variety of ways";

        var transformer = new Transformer("model-7b.bin");
        var tokens = transformer.Generate(" You are a helpful assistant. ", 19);

        var builder = new StringBuilder();
        foreach(var token in tokens)
            builder.Append(token);

        var actual = builder.ToString();

        Assert.Equal(expected, actual);
    }

    [Fact(Skip = "Requires tokenizer.bin")]
    public void Test_Tokenizer()
    {
        var tokenizer = new Tokenizer("tokenizer.bin", 32000);

        var actual = tokenizer.Encode(" You are a helpful assistant", true, true);

        var expected = new[] { 1, 887, 526, 263, 8444, 20255, 2 };

        Assert.Equal(expected, actual);

    }
}