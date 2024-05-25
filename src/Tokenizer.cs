using System.Text;
using System.Text.RegularExpressions;

namespace libLlama2;

public partial class Tokenizer : ITokenizer
{
    private const int BOS = 1; // Begining Of Sequence

    private const int EOS = 2; // End Of Sequence

    private struct TokenIndex : IComparable<TokenIndex>
    {
        public string str;
        public int id;
        public readonly int CompareTo(TokenIndex other) =>
            string.CompareOrdinal(str, other.str);
    }

    private readonly string[] vocab;

    private readonly TokenIndex[] sortedVocab;

    private readonly float[] vocabScores;

    private readonly int maxTokenLength;

    public Tokenizer(string tokenizerPath, int vocabSize)
    {
        vocab = new string[vocabSize];
        sortedVocab = new TokenIndex[vocabSize];
        vocabScores = new float[vocabSize];

        using var fs = File.OpenRead(tokenizerPath);
        using var reader = new BinaryReader(fs);

        maxTokenLength = reader.ReadInt32();

        for (int i = 0; i < vocabSize; i++)
        {
            vocabScores[i] = reader.ReadSingle();

            int len = reader.ReadInt32();
            var buffer = new byte[len];
            _ = reader.Read(buffer);
            var piece = Encoding.UTF8.GetString(buffer);
            vocab[i] = piece;
            sortedVocab[i] = new TokenIndex { id = i, str = piece };
        }

        Array.Sort(sortedVocab);
    }

    [GeneratedRegex("^<0x([0-9A-F]{2})>$")]
    private static partial Regex EncodedByteRegex();

    public string Decode(int prev, int token)
    {
        var piece = vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        piece = (prev == BOS && piece[0] == ' ') ? piece.TrimStart() : piece;

        var match = EncodedByteRegex().Match(piece);
        if (!match.Success)
            return piece;

        var value = match.Groups[1].Value;
        var byteValue = Convert.ToByte(value, 16);
        var charValue = Convert.ToChar(byteValue);
        return charValue.ToString();
    }

    private int StrLookup(string str)
    {
        var tok = new TokenIndex { str = str };
        var i = Array.BinarySearch(sortedVocab, tok);
        return i > 0 ? sortedVocab[i].id : -1;
    }

    public int[] Encode(string prompt, bool bos = false, bool eos = false)
    {
        var tokens = new List<int>(prompt.Length + 3);

        // first encode every individual byte in the input string

        if (bos) tokens.Add(BOS);

        if (!string.IsNullOrEmpty(prompt))
        {
            var dummy_prefix = StrLookup(" ");
            tokens.Add(dummy_prefix);
        }

        foreach (char c in prompt)
        {
            var id = StrLookup(c.ToString());
            if (id != -1) {
                tokens.Add(id);
            } else {
                var bytes = Encoding.UTF8.GetBytes(c.ToString());
                foreach(var b in bytes)
                    tokens.Add(b + 3);
            }
        }

        var strBuffer = new StringBuilder(maxTokenLength * 2 + 1); // *2 for concat, +1 for null terminator

        // merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while (true)
        {
            var bestScore = float.MinValue;
            var bestId = -1;
            var bestIdx = -1;

            for (var i = 0; i < tokens.Count - 1; i++)
            {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.Clear();
                strBuffer.Append(vocab[tokens[i]]);
                strBuffer.Append(vocab[tokens[i + 1]]);

                var id = StrLookup(strBuffer.ToString());
                if (id != -1 && vocabScores[id] > bestScore)
                {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id];
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
            tokens[bestIdx] = bestId;

            // delete token at position bestIdx+1, shift the entire sequence back 1
            tokens.RemoveAt(bestIdx + 1);
        }

        if (eos) tokens.Add(EOS);

        return tokens.ToArray();
    }

}