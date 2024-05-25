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

    public static IEnumerable<object[]> TestData()
    {
        yield return new object[] { "You are a helpful assistant", new[] { 1, 887, 526, 263, 8444, 20255, 2 } };
        yield return new object[] { "<<SYS>>", new[] { 1, 3532, 14816, 29903, 6778, 2 } };
        yield return new object[] { "[/INST]", new[] { 1, 518, 29914, 25580, 29962, 2 } };
        yield return new object[] { "\n", new[] { 1, 29871, 13, 2 } };
        yield return new object[] { "🦙", new[] { 1, 29871, 243, 162, 169, 156, 2 } };
    }

    [Theory(Skip = "Requires tokenizer.bin")]
    [MemberData(nameof(TestData))]
    public void Test_TokenizerEncode(string value, int[] expected)
    {
        var tokenizer = new Tokenizer("tokenizer.bin", 32000);

        var actual = tokenizer.Encode(value, true, true);

        Assert.Equal(expected, actual);

    }

    [Theory(Skip = "Requires tokenizer.bin")]
    [MemberData(nameof(TestData))]
    public void Test_TokenizerDecode(string test, int[] value)
    {
        var expected = $"\n<s>\n{test}\n</s>\n";
        var tokenizer = new Tokenizer("tokenizer.bin", 32000);

        var sb = new StringBuilder();
        var prev = 0;
        foreach (var token in value)
        {
            sb.Append(tokenizer.Decode(prev, token));
            prev = token;
        }
        var actual = sb.ToString();

        Assert.Equal(expected, actual);

    }
}