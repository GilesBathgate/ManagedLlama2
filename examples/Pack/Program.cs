namespace Example.Setup;

using libLlama2;

public class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 5)
        {
            Console.WriteLine("Usage: Program.exe <model-config.json> <quant-model.pt> <tokenizer.model> <output-model.bin> <output-tokenizer.bin>");
            return;
        }

        var modelConfig = args[0];
        var modelPath = args[1];
        var tokenizerPath = args[2];
        var modelOutputPath = args[3];
        var tokenizerOutputPath = args[4];

        using var modelConverter = new ModelConverter(
            modelPath,
            modelConfig,
            ModelConverter.ModelType.Llama2_AWQ_7b,
            download: false);

        modelConverter.Convert(modelOutputPath, ModelConverter.Format.v1);

        using var tokenizerConverter = new TokenizerConverter(
            tokenizerPath,
            download: false);

        tokenizerConverter.Convert(tokenizerOutputPath);
    }
}
