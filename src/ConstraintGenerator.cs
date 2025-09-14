namespace libLlama2;

using JsonState = JsonStateMachine.JsonState;

public class ConstraintGenerator
{
    public readonly struct Constraint
    {
        public bool Allowed { get; }

        public ISet<Token> Tokens { get; }

        public Constraint(bool allowed, ISet<Token> tokens)
        {
            Allowed = allowed;
            Tokens = tokens;
        }
    }

    private int vocabSize;

    private ITokenizer tokenizer;

    private IDictionary<(JsonState, JsonState), Constraint> precomputedConstraits;

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = vocabSize;
        precomputedConstraits = PrecomputeConstraints();
    }

    public ConstraintGenerator(ITokenizer tokenizer, int vocabSize, IDictionary<(JsonState, JsonState), Constraint> precomputed)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = vocabSize;
        precomputedConstraits = precomputed;
    }

    private IEnumerable<Token> AllTokens() =>
        Enumerable.Range(0, vocabSize).Select(x => new Token(x, tokenizer.Decode(x)));

    private bool IsTokenValidForState(JsonState state, JsonState context, Token token)
    {
        if (string.IsNullOrEmpty(token.Value)) return false;

        var tempMachine = new JsonStateMachine(tokenizer, state, context);
        tempMachine.Process(token.Value);
        return tempMachine.State != JsonState.Error;
    }

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

    public (bool allow, int[] tokens) ConstrainedTokens(JsonStateMachine stateMachine)
    {
        var key = (stateMachine.State, stateMachine.CurrentContext);
        var constraint = precomputedConstraits[key];
        return (constraint.Allowed, constraint.Tokens.Select(x => x.Id).ToArray());
    }

    public IDictionary<(JsonState, JsonState), Constraint> PrecomputeConstraints()
    {
        var result = new Dictionary<(JsonState, JsonState), Constraint>();

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
                    result[key] = new Constraint(true, stateValidTokens);
                }
                else
                {
                    result[key] = new Constraint(false, stateInvalidTokens);
                }
            }
        }
        return result;
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