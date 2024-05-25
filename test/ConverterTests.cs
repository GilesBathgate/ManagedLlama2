using System.Security.Cryptography;

namespace libLlama2.UnitTests;

public class ConverterTests
{
    private static byte[] ConvertModel(string outputPath)
    {
        var converter = new ModelConverter(download: true);
        using var fileStream = File.Create(outputPath);
        using var hash = SHA256.Create();
        using var hashStream = new CryptoStream(fileStream, hash, CryptoStreamMode.Write);
        converter.Convert(hashStream);
        hashStream.FlushFinalBlock();
        return hash.Hash!;
    }

    [Fact(Skip = "Slow, requires network")]
    public void ModelConverter_Test()
    {
        var expectedHash = "d398de0146c7c0e7d7087883421dd87b60f45288ce029cebb60208b0c0a65c91";

        var outputPath = "model-7b.bin";

        var hashBytes = ConvertModel(outputPath);

        var actualHash = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();

        Assert.Equal(expectedHash, actualHash);
    }

    [Fact(Skip = "Slow, requires network")]
    public void TokenizerConverter_Test()
    {
        var converter = new TokenizerConverter(download: true);
        converter.Convert("tokenizer.bin");
    }
}