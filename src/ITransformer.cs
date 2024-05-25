namespace libLlama2;

public interface ITransformer
{
    public IEnumerable<string> Generate(string prompt, int steps);

    public IEnumerable<string> Chat(string system_prompt, IEnumerable<string> userInput);

}
