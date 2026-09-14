namespace libLlama2;

public interface ITransformer
{
    public IEnumerable<Token> Generate(string prompt, int steps);

    public IEnumerable<Token> Chat(string system_prompt, IEnumerable<string> userInput);

}
