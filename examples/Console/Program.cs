namespace Example.Console;

using libLlama2;
using Console = System.Console;

public class Program
{
    const string systemPrompt = @"You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.";

    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: Program.exe <model.bin> <tokenizer.bin>");
            return;
        }

        var modelPath = args[0];
        var tokenizerPath = args[1];

        var transformer = new Transformer(modelPath, tokenizerPath);

        static IEnumerable<string> ReadInput()
        {
            while (true)
            {
                Console.Write("User: ");
                yield return Console.ReadLine() ?? string.Empty;
            }
        }

        var tokens = transformer.Chat(systemPrompt, ReadInput());

        foreach (var token in tokens)
            Console.Write(token);

        Console.WriteLine();
    }
}
