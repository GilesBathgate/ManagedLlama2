namespace libLlama2;

public class AnsiColorizer : IColorizer
{
    public string Colorize(Token token)
    {
        var value = token.Value.ReplaceLineEndings($"{EmptyColor}{Environment.NewLine}");

        var current = token.Id % standardColors.Length;
        if (current == previous)
        {
            previous = -1;
            return $"{alternateColors[current]}{value}";
        }
        previous = current;
        return $"{standardColors[current]}{value}";
    }

    private int previous = -1;

    public string EmptyColor { get => "\x1b[0m"; }

    private static readonly string[] standardColors =
    {
        "\x1b[41m", // Red
        "\x1b[42m", // Green
        "\x1b[43m", // Yellow
        "\x1b[44m", // Blue
        "\x1b[45m", // Magenta
        "\x1b[46m", // Cyan
    };

    private static readonly string[] alternateColors =
    {
        "\x1b[101m", // Red
        "\x1b[102m", // Green
        "\x1b[103m", // Yellow
        "\x1b[104m", // Blue
        "\x1b[105m", // Magenta
        "\x1b[106m", // Cyan
    };
}
