namespace libLlama2;

public class TokenizerDownloader : Downloader
{
    private const string tokenizerURL = "https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/tokenizer.model";

    private const string tokenizerHash = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347";

    private readonly string tokenizerPath;

    public TokenizerDownloader(string tokenizerPath) =>
        this.tokenizerPath = tokenizerPath;

    public override void Download()
    {
        Download(tokenizerURL, tokenizerPath, tokenizerHash, TimeSpan.FromMilliseconds(10));
    }

}