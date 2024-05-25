using System.Text;
using System.Text.RegularExpressions;
using ProtoBuf;

namespace libLlama2;

public partial class TokenizerConverter
{
    private readonly string tokenizerPath;

    public TokenizerConverter(bool download) : this("tokenizer.spm", download)
    {
    }

    public TokenizerConverter(string tokenizerPath, bool download = false)
    {
        if (!File.Exists(tokenizerPath))
        {
            if (!download) throw new ArgumentException("Download set to false and tokeniser path does not exist");
            Downloader.DownloadDefaultTokeniser(tokenizerPath);
        }

        this.tokenizerPath = tokenizerPath;
    }

    public void Convert(string outputPath)
    {
        using var fileStream = File.OpenRead(tokenizerPath);
        var model = Serializer.Deserialize<ModelProto>(fileStream);

        var parts = new List<(byte[] token, float score)>();
        foreach (var part in model.Pieces)
        {
            var piece = ConvertSpecial(part.Piece);
            piece = piece.Replace("▁", " ");

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

    [GeneratedRegex("^<0x?([0-9a-fA-F]{2})>$")]
    private static partial Regex EncodedByteRegex();

    public static string ConvertSpecial(string piece)
    {
        if (piece == "<s>")
            return "\n<s>\n";
        else if (piece == "</s>")
            return "\n</s>\n";

        var match = EncodedByteRegex().Match(piece);
        if (!match.Success)
            return piece;

        var value = match.Groups[1].Value;
        var byteValue = System.Convert.ToByte(value, 16);
        var charValue = System.Convert.ToChar(byteValue);

        return charValue.ToString();
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