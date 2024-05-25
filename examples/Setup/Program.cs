namespace Example.Setup;

using libLlama2;

public class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: Program.exe <output-model.bin> <output-tokenizer.bin>");
            return;
        }

        var modelPath = args[0];
        var tokenizerPath = args[1];

        using var modelConverter = new ModelConverter(ModelConverter.ModelType.Llama2_AWQ_13b, download: true);
        modelConverter.Convert(modelPath);

        using var tokenizerConverter = new TokenizerConverter(download: true);
        tokenizerConverter.Convert(tokenizerPath);

    }
}
