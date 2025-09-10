namespace libLlama2;

public class InertColorizer : IColorizer
{
    public string EmptyColor =>
        string.Empty;

    public string Colorize(Token token) =>
        token.Value;
}