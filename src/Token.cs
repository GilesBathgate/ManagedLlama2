namespace libLlama2;

public class Token
{
    public int Id { get; }

    public string Value { get; }

    public Token(int id, string value)
    {
        Id = id;
        Value = value;
    }

    public override string ToString() =>
        Value;

    public static implicit operator string(Token source) =>
        source.ToString();

    public override bool Equals(object? obj) =>
         (obj is Token other) && Id == other.Id && Value == other.Value;

    public override int GetHashCode() => Id;
}