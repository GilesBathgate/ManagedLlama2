namespace libLlama2;

public interface ITokenizer
{
    public string Decode(int token);

    public string Decode(int prev, int token);

    public int[] Encode(string prompt, bool bos = false, bool eos = false);

}