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

        var modelConverter = new ModelConverter(download: true);
        modelConverter.Convert(modelPath);

        var tokenizerConverter = new TokenizerConverter(download: true);
        tokenizerConverter.Convert(tokenizerPath);

        File.Delete("pytorch_model.pt");
        File.Delete("config.json");
        File.Delete("tokenizer.spm");
    }
}
