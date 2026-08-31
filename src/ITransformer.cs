namespace libLlama2;

public interface ITransformer
{
    public int Position { get; }

    public void Rollback(int position);

    public IEnumerable<Token> Generate(string prompt, int steps);

    public IEnumerable<Token> Chat(string system_prompt, IEnumerable<string> userInput);

    public IEnumerable<Token> RollbackToTurn(int turnPosition, string system_prompt, IEnumerable<string> userInput);
}
