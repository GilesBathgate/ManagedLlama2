namespace libLlama2;

public class ConstraintGenerator
{
    private int vocabSize;

    private ITokenizer tokenizer;

    private List<Token> constraintTokens = new();

    private IDictionary<ITransition, Constraint> constraints;

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize, IStateMachine stateMachine) : this(tokenizer, vocabSize, new Dictionary<ITransition, Constraint>())
    {
        PrecomputeConstraints(stateMachine);
    }

    private ConstraintGenerator(ITokenizer tokenizer, int vocabSize, IDictionary<ITransition, Constraint> constraints)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = vocabSize;
        this.constraints = constraints;
    }

    public IEnumerable<int> AllConstraints
    {
        get => constraintTokens.Select(x => x.Id);
    }

    public Constraint? CurrentConstraint(IStateMachine stateMachine)
    {
        if (constraints.TryGetValue(stateMachine.Transition, out var constraint))
            return constraint;

        return null;
    }

    private IEnumerable<Token> AllTokens() =>
        Enumerable.Range(0, vocabSize).Select(x => new Token(x, tokenizer.Decode(x)));

    private void PrecomputeConstraints(IStateMachine stateMachine)
    {
        foreach (var transition in stateMachine.PossibleTransitions())
        {
            PrecomputeConstraints(transition);
        }
    }

    private void PrecomputeConstraints(ITransition transition)
    {
        var stateValidTokens = new HashSet<Token>();
        var stateInvalidTokens = new HashSet<Token>();

        foreach (var token in AllTokens())
        {
            if (transition.CanBeFollowedBy(token))
            {
                stateValidTokens.Add(token);
            }
            else
            {
                stateInvalidTokens.Add(token);
            }
        }


        if (stateValidTokens.Count <= stateInvalidTokens.Count)
        {
            constraints[transition] = new Constraint(true, constraintTokens.Count, stateValidTokens.Count);
            constraintTokens.AddRange(stateValidTokens);
        }
        else
        {
            constraints[transition] = new Constraint(false, constraintTokens.Count, stateInvalidTokens.Count);
            constraintTokens.AddRange(stateInvalidTokens);
        }
    }

}