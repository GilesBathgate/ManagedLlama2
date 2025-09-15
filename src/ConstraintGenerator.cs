namespace libLlama2;

using JsonState = JsonStateMachine.JsonState;

public class ConstraintGenerator
{
    private int vocabSize;

    private ITokenizer tokenizer;

    private List<Token> constraintTokens = new();

    private IDictionary<(JsonState, JsonState), Constraint> constraints;

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize) : this(tokenizer, vocabSize, new Dictionary<(JsonState, JsonState), Constraint>())
    {
        PrecomputeConstraints();
    }

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize, IDictionary<(JsonState, JsonState), Constraint> constraints)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = vocabSize;
        this.constraints = constraints;
    }

    public IEnumerable<int> AllConstraints
    {
        get => constraintTokens.Select(x => x.Id);
    }

    private IEnumerable<Token> AllTokens() =>
        Enumerable.Range(0, vocabSize).Select(x => new Token(x, tokenizer.Decode(x)));

    private bool IsTokenValidForState(JsonState state, JsonState context, Token token)
    {
        if (string.IsNullOrEmpty(token.Value)) return false;

        var tempMachine = new JsonStateMachine(state, context);
        tempMachine.Process(token.Value);
        return tempMachine.State != JsonState.Error;
    }

    [Obsolete]
    public IDictionary<JsonState, ISet<Token>> PrecomputeValidTokens(JsonState context)
    {
        var validTokens = new Dictionary<JsonState, ISet<Token>>();
        foreach (JsonState state in Enum.GetValues(typeof(JsonState)))
        {
            var stateValidTokens = new HashSet<Token>();
            foreach (var token in AllTokens())
            {
                if (IsTokenValidForState(state, context, token))
                    stateValidTokens.Add(token);
            }
            validTokens[state] = stateValidTokens;
        }
        return validTokens;
    }

    [Obsolete]
    public IDictionary<(JsonState, JsonState), ISet<Token>> PrecomputeValidTokens()
    {
        var validTokens = new Dictionary<(JsonState, JsonState), ISet<Token>>();
        foreach (JsonState state in Enum.GetValues(typeof(JsonState)))
        {
            var contexts = PossibleContextsForState(state);
            foreach (var contextState in contexts)
            {
                var key = (state, contextState);
                var stateValidTokens = new HashSet<Token>();

                foreach (var token in AllTokens())
                {
                    if (IsTokenValidForState(state, contextState, token))
                    {
                        stateValidTokens.Add(token);
                    }
                }
                validTokens[key] = stateValidTokens;
            }
        }
        return validTokens;
    }

    public Constraint? CurrentConstraint(JsonStateMachine stateMachine)
    {
        var key = (stateMachine.State, stateMachine.CurrentContext);
        if (constraints.TryGetValue(key, out var constraint))
            return constraint;

        return null;
    }

    [Obsolete]
    public IList<Token> CurrentConstraintTokens(Constraint constraint)
    {
        return constraintTokens.Skip(constraint.Index).Take(constraint.Size).ToList();
    }

    private void PrecomputeConstraints()
    {
        foreach (JsonState state in Enum.GetValues(typeof(JsonState)))
        {
            var contexts = PossibleContextsForState(state);
            foreach (var contextState in contexts)
            {
                var key = (state, contextState);
                var stateValidTokens = new HashSet<Token>();
                var stateInvalidTokens = new HashSet<Token>();

                foreach (var token in AllTokens())
                {
                    if (IsTokenValidForState(state, contextState, token))
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
                    constraints[key] = new Constraint(true, constraintTokens.Count, stateValidTokens.Count);
                    constraintTokens.AddRange(stateValidTokens);
                }
                else
                {
                    constraints[key] = new Constraint(false, constraintTokens.Count, stateInvalidTokens.Count);
                    constraintTokens.AddRange(stateInvalidTokens);
                }
            }
        }
    }

    private IList<JsonState> PossibleContextsForState(JsonState state)
    {
        var contexts = new List<JsonState>();
        switch (state)
        {
            case JsonState.ExpectingCommaOrEnd:
            case JsonState.InString:
            case JsonState.InNumber:
            case JsonState.InBoolean:
            case JsonState.InNull:
                contexts.Add(JsonState.InObject);
                contexts.Add(JsonState.InArray);
                break;
            case JsonState.ExpectingKey:
            case JsonState.ExpectingColon:
            case JsonState.InStringKey:
            case JsonState.ExpectingFirstKey:
                contexts.Add(JsonState.InObject);
                break;
            case JsonState.ExpectingValue:
                contexts.Add(JsonState.InObject);
                contexts.Add(JsonState.InArray);
                contexts.Add(default);
                break;
            case JsonState.ExpectingFirstValue:
                contexts.Add(JsonState.InArray);
                break;
            case JsonState.InStringEscaped:
                contexts.Add(JsonState.InString);
                contexts.Add(JsonState.InStringKey);
                break;
            default:
                contexts.Add(default);
                break;
        }
        return contexts;
    }

}