namespace libLlama2;

using JsonTransition = JsonStateMachine.JsonTransition;

public class ConstraintGenerator
{
    private int vocabSize;

    private ITokenizer tokenizer;

    private List<Token> constraintTokens = new();

    private IDictionary<JsonTransition, Constraint> constraints;

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize, JsonStateMachine stateMachine) : this(tokenizer, vocabSize, new Dictionary<JsonTransition, Constraint>())
    {
        PrecomputeConstraints(stateMachine);
    }

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize, IDictionary<JsonTransition, Constraint> constraints)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = vocabSize;
        this.constraints = constraints;
    }

    public IEnumerable<int> AllConstraints
    {
        get => constraintTokens.Select(x => x.Id);
    }

    public Constraint? CurrentConstraint(JsonStateMachine stateMachine)
    {
        if (constraints.TryGetValue(stateMachine.Transition, out var constraint))
            return constraint;

        return null;
    }

    private IEnumerable<Token> AllTokens() =>
        Enumerable.Range(0, vocabSize).Select(x => new Token(x, tokenizer.Decode(x)));

    private bool IsTokenValidForTransition(JsonTransition transition, Token token)
    {
        if (string.IsNullOrEmpty(token.Value)) return false;

        var tempMachine = new JsonStateMachine(transition);
        tempMachine.Process(token.Value);
        return tempMachine.IsValid;
    }

    private void PrecomputeConstraints(JsonStateMachine stateMachine)
    {
        foreach (var transition in stateMachine.PossibleTransitions())
        {
            PrecomputeConstraints(transition);
        }
    }

    private void PrecomputeConstraints(JsonTransition transition)
    {
        var stateValidTokens = new HashSet<Token>();
        var stateInvalidTokens = new HashSet<Token>();

        foreach (var token in AllTokens())
        {
            if (IsTokenValidForTransition(transition, token))
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