using System.Security.Cryptography;

namespace libLlama2;

public abstract class Downloader
{

    public abstract void Download();

    protected static void Download(string url, string path, string expectedHash, TimeSpan waitTime)
    {
        Console.WriteLine($"Downloading {path}");
        var downloadTask = DownloadAsync(url, path, expectedHash);
        while (!downloadTask.IsCompleted)
        {
            downloadTask.Wait(waitTime);
            Console.Write(".");
        }
        Console.WriteLine("done");
    }

    protected static async Task DownloadAsync(string url, string path, string expectedHash)
    {
        using var client = new HttpClient();
        using var stream = await client.GetStreamAsync(url);
        using var fileStream = File.OpenWrite(path);

        using var hash = SHA256.Create();
        using var hashStream = new CryptoStream(stream, hash, CryptoStreamMode.Read);
        await hashStream.CopyToAsync(fileStream);

        try
        {
            if (hash.Hash is null)
                throw new InvalidDataException("Invalid hash");
            var hashString = BitConverter.ToString(hash.Hash).Replace("-", "").ToLowerInvariant();
            if (hashString != expectedHash)
                throw new InvalidDataException($"Incorrect hash {hashString} != {expectedHash}");
        }
        catch (InvalidDataException)
        {
            fileStream.Close();
            File.Delete(path);
            throw;
        }
    }
}