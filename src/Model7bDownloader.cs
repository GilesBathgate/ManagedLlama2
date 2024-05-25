namespace libLlama2;

public class Model7bDownloader : Downloader
{
    private const string modelUrl = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/pytorch_model.bin";

    private const string modelHash = "d4b37309c09f20316215c98bdf1205070718ae2c0fccd25d4c9e799cc84d7a65";

    private const string configUrl = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/config.json";

    private const string configHash = "b81849ea66286f37ddf52d7f49f2ba46a21fa27a2df0149410adfa62984ee16e";

    private readonly string modelPath;

    private readonly string configPath;

    public Model7bDownloader(string modelPath, string configPath) =>
        (this.modelPath, this.configPath) = (modelPath, configPath);

    public override void Download()
    {
        Download(modelUrl, modelPath, modelHash, TimeSpan.FromSeconds(1));
        Download(configUrl, configPath, configHash, TimeSpan.FromMilliseconds(5));
    }

}