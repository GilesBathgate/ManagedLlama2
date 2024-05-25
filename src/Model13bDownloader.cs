namespace libLlama2;

public class Model13bDownloader : Downloader
{

    private const string modelUrl1 = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00001-of-00003.bin";

    private const string modelUrl2 = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00002-of-00003.bin";

    private const string modelUrl3 = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00003-of-00003.bin";

    private const string modelHash1 = "64723df7c76ee78cb03387ea6cf19144dd48f1187e4e061decffe12d80681b7d";

    private const string modelHash2 = "405920fc23b0ac8c0c4d741c1b8e651e99bdf485866d70d17d108658327d3e60";

    private const string modelHash3 = "3b2758b03c1996c49adc839d14c5cc5727c4110af6739e61ad0aea1935ae17fe";

    private const string configUrl = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/config.json";

    private const string configHash = "280a96db75813fa467c4940820a9efe11345fe1b00739a53a77ac462c7357afc";

    private readonly IList<string> modelPaths;

    private readonly string configPath;

    public Model13bDownloader(IList<string> modelPaths, string configPath) =>
        (this.modelPaths, this.configPath) = (modelPaths, configPath);

    public override void Download()
    {
        var modelUrls = new[] { modelUrl1, modelUrl2, modelUrl3 };
        var modelHashes = new[] { modelHash1, modelHash2, modelHash3 };
        foreach (var (i, path) in modelPaths.Enumerate())
            Download(modelUrls[i], path, modelHashes[i], TimeSpan.FromSeconds(1));

        Download(configUrl, configPath, configHash, TimeSpan.FromMilliseconds(5));
    }

}