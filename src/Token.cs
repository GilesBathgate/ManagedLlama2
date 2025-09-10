namespace libLlama2;

public class Token
{
    public int Id { get; }

    public string Value { get; }

    internal Token(int id, string value)
    {
        Id = id;
        Value = value;
    }

    public override string ToString() =>
        Value;

    public static implicit operator string(Token source) =>
        source.ToString();
}