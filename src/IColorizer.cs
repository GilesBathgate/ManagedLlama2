namespace libLlama2;

public interface IColorizer
{
    public string EmptyColor { get; }
    public string Colorize(Token token);
}
