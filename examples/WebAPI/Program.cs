namespace Example.WebAPI;

using libLlama2;
using System.Text;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Collections.Concurrent;

public class Server
{
    const string hostname = "localhost";

    const int port = 5000;

    private readonly BlockingCollection<HttpListenerContext> requestQueue = new();

    public Server(string modelPath, string tokenizerPath) =>
        Task.Run(() => ProcessRequests(modelPath, tokenizerPath));

    private (string prompt, int maxTokens) ReadRequest(HttpListenerContext context)
    {
        var request = context.Request;
        var jsonDocument = JsonDocument.Parse(request.InputStream);
        var jsonElement = jsonDocument.RootElement;

        var prompt = jsonElement.GetProperty("prompt").GetString();
        if (string.IsNullOrWhiteSpace(prompt)) throw new InvalidOperationException("Empty prompt");

        var maxTokens = jsonElement.GetProperty("max_tokens").GetInt32();
        if (maxTokens <= 0) throw new InvalidOperationException("Invalid max tokens");

        return (prompt, maxTokens);
    }

    private void WriteResponse(HttpListenerContext context, string result, string type)
    {
        using var response = context.Response;
        response.ContentType = "application/json";
        using var writer = new Utf8JsonWriter(response.OutputStream);
        writer.WriteStartObject();
        writer.WritePropertyName(type);
        writer.WriteStringValue(result);
        writer.WriteEndObject();
    }

    private void ProcessRequests(string modelPath, string tokenizerPath)
    {
        var transformer = new Transformer(modelPath, tokenizerPath);

        while (true)
        {
            var context = requestQueue.Take();

            try
            {
                var (prompt, maxTokens) = ReadRequest(context);

                var result = string.Join("", transformer.Generate(prompt, maxTokens));

                WriteResponse(context, result, "text");
            }
            catch (Exception e)
            {
                // Log exception etc..
                WriteResponse(context, e.Message, "error");
            }
        }
    }

    public async Task Start()
    {
        var listener = new HttpListener();
        var prefix = $"http://{hostname}:{port}/";
        listener.Prefixes.Add(prefix);
        listener.Start();
        Console.WriteLine($"Running on {prefix}");

        while (true)
        {
            var context = await listener.GetContextAsync();
            var request = context.Request;
            if (request.HttpMethod == HttpMethod.Post.Method)
                requestQueue.Add(context);
        }
    }
}

public class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: Program.exe <model.bin> <tokenizer.bin>");
            return;
        }

        var modelPath = args[0];
        var tokenizerPath = args[1];

        var server = new Server(modelPath, tokenizerPath);
        await server.Start();
    }
}
