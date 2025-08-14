using System.Text;
using ProtoBuf;

namespace libLlama2;

public class TokenizerConverter : IDisposable
{
    private readonly Downloader downloader;
    private readonly string tokenizerPath;

    public TokenizerConverter(bool download) : this("tokenizer.spm", download)
    {
    }

    public TokenizerConverter(string tokenizerPath, bool download = false)
    {
        if (download)
        {
            downloader = new TokenizerDownloader(tokenizerPath);
            downloader.Download();
        }
        else
        {
            if (!File.Exists(tokenizerPath))
                throw new ArgumentException("Download set to false and tokeniser path does not exist");
        }

        this.tokenizerPath = tokenizerPath;
    }

    public void Dispose()
    {
        if (downloader is null) return;

        File.Delete(tokenizerPath);
    }

    public void Convert(string outputPath)
    {
        using var fileStream = File.OpenRead(tokenizerPath);
        var model = Serializer.Deserialize<ModelProto>(fileStream);

        var parts = new List<(byte[] token, float score)>();
        foreach (var part in model.Pieces)
        {
            var piece = ConvertPiece(part.Piece);

            var token = Encoding.UTF8.GetBytes(piece);
            parts.Add((token, part.Score));
        }

        var maxTokenLength = parts.Max(x => x.token.Length);

        using var outputStream = File.Create(outputPath);
        using var writer = new BinaryWriter(outputStream);
        writer.Write(maxTokenLength);

        foreach (var (token, score) in parts)
        {
            writer.Write(score);
            writer.Write(token.Length);
            writer.Write(token);
        }

    }

    public static string ConvertPiece(string piece)
    {
        if (piece == "<s>")
            return "\n<s>\n";
        else if (piece == "</s>")
            return "\n</s>\n";

        return piece.Replace("▁", " ");
    }

    [ProtoContract]
    public class ModelProto
    {
        [ProtoMember(1, Name = "pieces")]
        public IList<SentencePiece> Pieces { get; } = new List<SentencePiece>();

        [ProtoContract]
        public class SentencePiece
        {
            [ProtoMember(1, Name = "piece")]
            public required string Piece;

            [ProtoMember(2, Name = "score")]
            public float Score;
        }
    }

}