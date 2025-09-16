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
        InStringEscaped,
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

    public class JsonTransition
    {
        public JsonState State { get; }

        public JsonState Context { get; }

        public JsonTransition(JsonState state, JsonState context)
        {
            State = state;
            Context = context;
        }

        public override bool Equals(object? obj)
        {
            if (obj is JsonTransition other)
                return other.State == State && other.Context == Context;

            return false;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Context, State);
        }
    }

    private JsonState State { get; set; } = JsonState.Initial;

    private JsonState CurrentContext => context.Count > 0 ? context.Peek() : JsonState.Initial;

    public JsonTransition Transition => new JsonTransition(State, CurrentContext);

    private readonly Stack<JsonState> context = new();

    private readonly StringBuilder buffer = new();

    public delegate void StateChangedHandler(JsonState newState);

    public event StateChangedHandler? StateChanged;

    public JsonStateMachine()
    {
    }

    public JsonStateMachine(JsonTransition transition)
    {
        this.State = transition.State;
        this.context = new Stack<JsonState>([transition.Context]);
    }

    public bool IsValid { get => State != JsonState.Error; }
    public bool IsComplete { get => State == JsonState.Complete; }

    public void Reset()
    {
        context.Clear();
        buffer.Clear();
        State = JsonState.Initial;
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
            case JsonState.InStringEscaped: return HandleInStringEscaped(c);
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
        if (c == '\\') { context.Push(State); ChangeState(JsonState.InStringEscaped); }
        else if (c == '"') ChangeState(JsonState.ExpectingCommaOrEnd);
        else if (c == '\n' || c == '\r') ChangeState(JsonState.Error);
        else buffer.Append(c);
        return true;
    }

    private bool HandleInStringKey(char c)
    {
        if (c == '\\') { context.Push(State); ChangeState(JsonState.InStringEscaped); }
        else if (c == '"') ChangeState(JsonState.ExpectingColon);
        else if (c == '\n' || c == '\r') ChangeState(JsonState.Error);
        else buffer.Append(c);
        return true;
    }

    private bool HandleInStringEscaped(char c)
    {
        if ("\"\\/bfnrt".Contains(c))
        {
            buffer.Append(c);
            ChangeState(context.Pop());
        }
        else
        {
            ChangeState(JsonState.Error);
        }
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

    public IEnumerable<JsonTransition> PossibleTransitions()
    {
        foreach (JsonState state in Enum.GetValues(typeof(JsonState)))
        {
            var contexts = PossibleContextsForState(state);
            foreach (var context in contexts)
            {
                yield return new JsonTransition(state, context);
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
                contexts.Add(JsonState.Initial);
                break;
            case JsonState.ExpectingFirstValue:
                contexts.Add(JsonState.InArray);
                break;
            case JsonState.InStringEscaped:
                contexts.Add(JsonState.InString);
                contexts.Add(JsonState.InStringKey);
                break;
            default:
                contexts.Add(JsonState.Initial);
                break;
        }
        return contexts;
    }

}
