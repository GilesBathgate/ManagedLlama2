using System.Text;

namespace libLlama2;

public class JsonStateMachine
{
    public enum JsonState
    {
        Initial,
        InObject,
        InArray,
        InString,
        InStringKey,
        InNumber,
        InBoolean,
        InNull,
        ExpectingValue,
        ExpectingFirstValue,
        ExpectingKey,
        ExpectingFirstKey,
        ExpectingColon,
        ExpectingCommaOrEnd,
        Complete,
        Error
    }

    private readonly Config config;

    private readonly ITokenizer tokenizer;

    public JsonState State { get; private set; } = JsonState.Initial;

    public JsonState CurrentContext => context.Count > 0 ? context.Peek() : default;

    private readonly Stack<JsonState> context = new();

    private readonly StringBuilder buffer = new();

    public delegate void StateChangedHandler(JsonState newState);

    public event StateChangedHandler? StateChanged;

    private IDictionary<(JsonState, JsonState), ISet<Token>> precomputedValidTokens;

    public JsonStateMachine(ITokenizer tokenizer, Config config)
    {
        this.tokenizer = tokenizer;
        this.config = config;
        precomputedValidTokens = PrecomputeValidTokens();
    }

    internal JsonStateMachine(ITokenizer tokenizer, Config config, JsonState initialState, JsonState initialContextState)
    {
        this.tokenizer = tokenizer;
        this.config = config;
        this.State = initialState;
        this.context = new Stack<JsonState>([initialContextState]);
        precomputedValidTokens = new Dictionary<(JsonState, JsonState), ISet<Token>>();
    }

    public void Reset()
    {
        context.Clear();
        buffer.Clear();
        State = JsonState.Initial;
    }

    public int[] ValidTokens()
    {
        var key = (State, CurrentContext);
        return precomputedValidTokens[key].Select(x => x.Id).ToArray();
    }

    public void Process(Token token) =>
        Process(token.Value);

    public void Process(string characters)
    {
        int i = 0;
        while (i < characters.Length)
        {
            if (Process(characters[i]))
            {
                i++;
            }
        }
    }

    public void Complete()
    {
        if (State != JsonState.Complete)
            ChangeState(JsonState.Error);
    }

    private bool Process(char c)
    {
        if (State == JsonState.Complete)
        {
            ChangeState(JsonState.Error);
            return true;
        }

        if (char.IsWhiteSpace(c) &&
            State != JsonState.InString &&
            State != JsonState.InStringKey &&
            State != JsonState.InNumber &&
            State != JsonState.InBoolean &&
            State != JsonState.InNull)
        {
            return true;
        }

        switch (State)
        {
            case JsonState.Initial: return HandleInitial(c);
            case JsonState.ExpectingKey: return HandleExpectingKey(c);
            case JsonState.ExpectingFirstKey: return HandleExpectingFirstKey(c);
            case JsonState.ExpectingColon: return HandleExpectingColon(c);
            case JsonState.ExpectingValue: return HandleExpectingValue(c);
            case JsonState.ExpectingFirstValue: return HandleExpectingFirstValue(c);
            case JsonState.ExpectingCommaOrEnd: return HandleExpectingCommaOrEnd(c);
            case JsonState.InString: return HandleInString(c);
            case JsonState.InStringKey: return HandleInStringKey(c);
            case JsonState.InNumber: return HandleInNumber(c);
            case JsonState.InBoolean: return HandleInBoolean(c);
            case JsonState.InNull: return HandleInNull(c);
            case JsonState.Error: return true;
        }
        return true;
    }

    private bool HandleInitial(char c)
    {
        if (c == '{') { context.Push(JsonState.InObject); ChangeState(JsonState.ExpectingFirstKey); }
        else if (c == '[') { context.Push(JsonState.InArray); ChangeState(JsonState.ExpectingFirstValue); }
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleExpectingFirstValue(char c)
    {
        if (c == ']')
        {
            if (context.Count > 0 && context.Peek() == JsonState.InArray)
            {
                context.Pop();
                ChangeState(context.Count == 0 ? JsonState.Complete : JsonState.ExpectingCommaOrEnd);
            }
            else
            {
                ChangeState(JsonState.Error);
            }
        }
        else
        {
            buffer.Clear();
            if (c == '"') { ChangeState(JsonState.InString); }
            else if (c == '{') { context.Push(JsonState.InObject); ChangeState(JsonState.ExpectingFirstKey); }
            else if (c == '[') { context.Push(JsonState.InArray); ChangeState(JsonState.ExpectingFirstValue); }
            else if (c == 't' || c == 'f') { buffer.Append(c); ChangeState(JsonState.InBoolean); }
            else if (c == 'n') { buffer.Append(c); ChangeState(JsonState.InNull); }
            else if (char.IsDigit(c) || c == '-') { buffer.Append(c); ChangeState(JsonState.InNumber); }
            else { ChangeState(JsonState.Error); }
        }
        return true;
    }

    private bool HandleExpectingFirstKey(char c)
    {
        if (c == '"') { buffer.Clear(); ChangeState(JsonState.InStringKey); }
        else if (c == '}')
        {
            if (context.Count > 0 && context.Peek() == JsonState.InObject)
            {
                context.Pop();
                ChangeState(context.Count == 0 ? JsonState.Complete : JsonState.ExpectingCommaOrEnd);
            }
            else
            {
                ChangeState(JsonState.Error);
            }
        }
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleExpectingKey(char c)
    {
        if (c == '"') { buffer.Clear(); ChangeState(JsonState.InStringKey); }
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleExpectingColon(char c)
    {
        if (c == ':') ChangeState(JsonState.ExpectingValue);
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleExpectingValue(char c)
    {
        buffer.Clear();
        if (c == '"') ChangeState(JsonState.InString);
        else if (c == '{') { context.Push(JsonState.InObject); ChangeState(JsonState.ExpectingFirstKey); }
        else if (c == '[') { context.Push(JsonState.InArray); ChangeState(JsonState.ExpectingFirstValue); }
        else if (c == 't' || c == 'f') { buffer.Append(c); ChangeState(JsonState.InBoolean); }
        else if (c == 'n') { buffer.Append(c); ChangeState(JsonState.InNull); }
        else if (char.IsDigit(c) || c == '-') { buffer.Append(c); ChangeState(JsonState.InNumber); }
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleExpectingCommaOrEnd(char c)
    {
        if (c == ',')
        {
            if (context.Count > 0 && context.Peek() == JsonState.InObject) ChangeState(JsonState.ExpectingKey);
            else if (context.Count > 0 && context.Peek() == JsonState.InArray) ChangeState(JsonState.ExpectingValue);
            else ChangeState(JsonState.Error);
        }
        else if ((c == '}' && context.Count > 0 && context.Peek() == JsonState.InObject) || (c == ']' && context.Count > 0 && context.Peek() == JsonState.InArray))
        {
            context.Pop();
            ChangeState(context.Count == 0 ? JsonState.Complete : JsonState.ExpectingCommaOrEnd);
        }
        else ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleInString(char c)
    {
        if (c == '"') ChangeState(JsonState.ExpectingCommaOrEnd);
        else if (c == '\n' || c == '\r') ChangeState(JsonState.Error);
        else buffer.Append(c);
        return true;
    }

    private bool HandleInStringKey(char c)
    {
        if (c == '"') ChangeState(JsonState.ExpectingColon);
        else if (c == '\n' || c == '\r') ChangeState(JsonState.Error);
        else buffer.Append(c);
        return true;
    }

    private bool HandleInNumber(char c)
    {
        if (char.IsDigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
        {
            buffer.Append(c);
            return true;
        }
        else
        {
            buffer.Clear();
            ChangeState(JsonState.ExpectingCommaOrEnd);
            return false;
        }
    }

    private bool HandleInBoolean(char c)
    {
        buffer.Append(c);
        string s = buffer.ToString();
        if (s == "true" || s == "false") { buffer.Clear(); ChangeState(JsonState.ExpectingCommaOrEnd); }
        else if (!"true".StartsWith(s) && !"false".StartsWith(s)) ChangeState(JsonState.Error);
        return true;
    }

    private bool HandleInNull(char c)
    {
        buffer.Append(c);
        string s = buffer.ToString();
        if (s == "null") { buffer.Clear(); ChangeState(JsonState.ExpectingCommaOrEnd); }
        else if (!"null".StartsWith(s)) ChangeState(JsonState.Error);
        return true;
    }

    private void ChangeState(JsonState newState)
    {
        if (State == newState) return;
        State = newState;
        StateChanged?.Invoke(newState);
    }

    private IEnumerable<Token> AllTokens() =>
        Enumerable.Range(0, config.vocabSize).Select(x => new Token(x, tokenizer.Decode(x)));

    private bool IsTokenValidForState(JsonState state, JsonState context, Token token)
    {
        if (string.IsNullOrEmpty(token.Value)) return false;

        var tempMachine = new JsonStateMachine(tokenizer, config, state, context);
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
            default:
                contexts.Add(default);
                break;
        }
        return contexts;
    }

}
