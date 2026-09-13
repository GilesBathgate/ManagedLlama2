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
                    let currentPosition = 0;

                    let pendingUserMsg = null;

                    function send() {{
                        const text = textarea.value;
                        if (!text) return;
                        const p = document.createElement('p');
                        p.className = 'user';
                        p.innerHTML = `&#x1F464; <span class='msg-text'>${{text}}</span> <button class='edit-btn' onclick='editTurn(this)'>Edit</button>`;
                        output.appendChild(p);
                        pendingUserMsg = p;

                        websocket.send(JSON.stringify({{ action: 'chat', text: text }}));
                        textarea.value = '';
                        avatar = true;
                    }}

                    window.editTurn = function(btn) {{
                        const p = btn.closest('.user');
                        if (p.dataset.position === undefined) return;
                        const pos = parseInt(p.dataset.position);
                        const oldText = p.querySelector('.msg-text').innerText;
                        const newText = prompt('Edit your message:', oldText);
                        if (newText === null || newText.trim() === '') return;

                        // Remove all subsequent elements in output
                        let node = p.nextSibling;
                        while (node) {{
                            const next = node.nextSibling;
                            node.remove();
                            node = next;
                        }}
                        p.querySelector('.msg-text').innerText = newText;
                        pendingUserMsg = p;

                        websocket.send(JSON.stringify({{ action: 'edit', position: pos, text: newText }}));
                        avatar = true;
                    }};

                    function write(data) {{
                        try {{
                            const msg = JSON.parse(data);
                            if (msg.type === 'start_turn') {{
                                currentPosition = msg.position;
                                if (pendingUserMsg) {
                                    pendingUserMsg.dataset.position = msg.position;
                                    pendingUserMsg = null;
                                }
                                return;
                            }}
                        }} catch (e) {{ }}

                        if (avatar) output.insertAdjacentHTML('beforeend', '&#x2728');
                        avatar = false;
                        output.insertAdjacentText('beforeend', data);
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

        var transformer = new Transformer(modelPath, tokenizerPath);
        var buffer = new byte[4096];

        while (webSocket.State == WebSocketState.Open)
        {
            var result = await webSocket.ReceiveAsync(buffer, CancellationToken.None);
            if (result.MessageType == WebSocketMessageType.Close)
                break;

            if (result.MessageType == WebSocketMessageType.Text)
            {
                var input = Encoding.UTF8.GetString(buffer, 0, result.Count);
                string userInput = input;
                int? rollbackPos = null;

                try
                {
                    using var doc = System.Text.Json.JsonDocument.Parse(input);
                    var root = doc.RootElement;
                    if (root.TryGetProperty("text", out var textElem))
                        userInput = textElem.GetString() ?? input;
                    if (root.TryGetProperty("action", out var actionElem) && actionElem.GetString() == "edit")
                    {
                        if (root.TryGetProperty("position", out var posElem))
                            rollbackPos = posElem.GetInt32();
                    }
                }
                catch
                {
                    userInput = input;
                }

                if (rollbackPos.HasValue)
                {
                    transformer.Rollback(rollbackPos.Value);
                }

                var startPosMsg = Encoding.UTF8.GetBytes($"{{\"type\":\"start_turn\",\"position\":{transformer.Position}}}");
                await webSocket.SendAsync(startPosMsg, WebSocketMessageType.Text, true, CancellationToken.None);

                var tokens = transformer.Chat(systemPrompt, new[] { userInput });
                foreach (var token in tokens)
                {
                    var tokenBytes = Encoding.UTF8.GetBytes(token);
                    await webSocket.SendAsync(tokenBytes, WebSocketMessageType.Text, true, CancellationToken.None);
                }
            }
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
