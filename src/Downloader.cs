using System.Security.Cryptography;

namespace libLlama2;

public class Downloader
{
    private const string modelUrl = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/pytorch_model.bin";

    private const string modelHash = "d4b37309c09f20316215c98bdf1205070718ae2c0fccd25d4c9e799cc84d7a65";

    private const string configUrl = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/config.json";

    private const string tokenizerURL = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/tokenizer.model";


    public static void DownloadDefaultModel(string modelPath) =>
        Download(modelUrl, modelPath, modelHash, TimeSpan.FromSeconds(1));

    public static void DownloadDefaultTokeniser(string tokenizerPath) =>
        Download(tokenizerURL, tokenizerPath, null, TimeSpan.FromMilliseconds(10));

    public static void DownloadDefaultConfig(string configPath) =>
        Download(configUrl, configPath, null, TimeSpan.FromMilliseconds(5));

    private static void Download(string url, string path, string? expectedHash, TimeSpan waitTime)
    {
        Console.WriteLine($"Downloading {path}");
        var downloadTask = DownloadAsync(url, path, expectedHash);
        while(!downloadTask.IsCompleted) {
            downloadTask.Wait(waitTime);
            Console.Write(".");
        }
        Console.WriteLine("done");
    }

    private static async Task DownloadAsync(string url, string path, string? expectedHash)
    {
        using var client = new HttpClient();
        using var stream = await client.GetStreamAsync(url);
        using var fileStream = File.OpenWrite(path);

        if (expectedHash is null) {
            await stream.CopyToAsync(fileStream);
            return;
        }

        using var hash = SHA256.Create();
        using var hashStream = new CryptoStream(stream, hash, CryptoStreamMode.Read);
        await hashStream.CopyToAsync(fileStream);

        try
        {
            if (hash.Hash is null)
                throw new InvalidDataException("Invalid hash");
            var hashString = BitConverter.ToString(hash.Hash).Replace("-","").ToLowerInvariant();
            if (hashString != expectedHash)
                throw new InvalidDataException($"Incorrect hash {hashString} != {expectedHash}");
        }
        catch(InvalidDataException)
        {
            fileStream.Close();
            File.Delete(path);
            throw;
        }
    }
}