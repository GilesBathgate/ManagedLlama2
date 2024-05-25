namespace Example.WebSocket;

using libLlama2;
using System.Net;
using System.Text;
using System.Net.WebSockets;

public class Program
{
    private const string hostname = "localhost";

    private const int port = 9292;

    private const string subProtocol = "example.websocket.chat";

    const string systemPrompt = @"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.";


    private static async Task ServeWebPage(HttpListenerContext context)
    {
        context.Response.ContentType = "text/html";
        string content = $@"
            <!doctype html>
            <html lang='en'>
                <head>
                    <style>
                        body {{ font-family: Arial, Helvetica, sans-serif; color: white; background-color: black; }}
                        .container {{ max-width: fit-content; margin: 0 auto; background-color: #141414; border-radius: 15px; padding: 10px; }}
                        #output {{ padding: 0 10px 10px 0; white-space: pre-wrap; width: 800px; height: calc(100vh - 100px); overflow-y: auto; scrollbar-color: #222222 #141414; line-height: 1.5; }}
                        textarea {{ color: white; width: 680px; vertical-align: bottom; }}
                        .user {{ background-color: #222222; border-radius: 15px; padding: 10px; }}
                        button {{ border-radius: 10px; height: 50px; width: 100px; color: white; background-color: #3b3b3b; }}
                    </style>
                </head>
                <body>
                    <div class='container'>
                        <div id='output'></div> <textarea class='user' rows='2'></textarea> <button>Send</button>
                    </div>
                </body>
                <script>
                    const output = document.querySelector('#output');
                    const textarea = document.querySelector('textarea');
                    const button = document.querySelector('button');
                    const origin = window.location;
                    const websocket = new WebSocket(`ws://${{origin.hostname}}:${{origin.port}}`, '{subProtocol}');
                    let avatar = false;
                    function send() {{
                        output.insertAdjacentHTML('beforeend', `<p class='user'>&#x1F464; ${{textarea.value}}</p>`);
                        websocket.send(textarea.value);
                        textarea.value = '';
                        avatar = true;
                    }}
                    function write(text) {{
                        if (avatar) output.insertAdjacentHTML('beforeend', '&#x2728');
                        avatar = false;
                        output.insertAdjacentText('beforeend', text);
                        output.scrollTop = output.scrollHeight;
                    }}
                    button.addEventListener('click', send);
                    websocket.onmessage = (e) => write(e.data);
                </script>
            </html>";
        using var writer = new StreamWriter(context.Response.OutputStream);
        await writer.WriteAsync(content);
    }

    static async Task ServeWebSocket(HttpListenerContext context, string modelPath, string tokenizerPath)
    {
        var webSocketContext = await context.AcceptWebSocketAsync(subProtocol);
        var webSocket = webSocketContext.WebSocket;

        static async IAsyncEnumerable<string> ReadInput(WebSocket webSocket)
        {
            var buffer = new byte[1024];
            while (true)
            {
                var response = await webSocket.ReceiveAsync(buffer, CancellationToken.None);
                if (response.MessageType == WebSocketMessageType.Text)
                    yield return Encoding.UTF8.GetString(buffer, 0, response.Count);
            }
        }

        var transformer = new Transformer(modelPath, tokenizerPath);
        var tokens = transformer.Chat(systemPrompt, ReadInput(webSocket).ToBlockingEnumerable());

        foreach (var token in tokens)
        {
            var buffer = Encoding.UTF8.GetBytes(token);
            await webSocket.SendAsync(buffer, WebSocketMessageType.Text, true, CancellationToken.None);
        }
    }

    static async Task Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: Program.exe <model.bin> <tokenizer.bin>");
            return;
        }

        var modelPath = args[0];
        var tokenizerPath = args[1];

        var listener = new HttpListener();
        var prefix = $"http://{hostname}:{port}/";
        listener.Prefixes.Add(prefix);
        listener.Start();
        Console.WriteLine($"Open browser on {prefix}");

        while (true)
        {
            var context = await listener.GetContextAsync();
            if (!context.Request.IsWebSocketRequest)
                await ServeWebPage(context);
            else
                await ServeWebSocket(context, modelPath, tokenizerPath);
        }
    }
}
