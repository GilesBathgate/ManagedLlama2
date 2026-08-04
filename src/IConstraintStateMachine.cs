namespace libLlama2;

public interface IConstraintStateMachine : IStateMachine
{
    void Process(Token token);
    void Reset();
    bool IsComplete { get; }
}
