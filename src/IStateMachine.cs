namespace libLlama2;

public interface IStateMachine
{
    ITransition Transition { get; }
    IEnumerable<ITransition> PossibleTransitions();
}