using JsonState = libLlama2.JsonStateMachine.JsonState;
using JsonTransition = libLlama2.JsonStateMachine.JsonTransition;

namespace libLlama2.UnitTests;

public class JsonStateMachineTests
{
    [Theory]
    [InlineData("{\"key\":\"value\"}")]
    [InlineData("[1, \"two\", true]")]
    [InlineData("{\"a\":{\"b\":1}}")]
    [InlineData("[1, 2, 3]")]
    [InlineData("[]")]
    [InlineData("{}")]
    [InlineData("{\"a\":[]}")]
    [InlineData("[{}]")]
    [InlineData("{\"a\":1, \"b\":\"hello\", \"c\":true, \"d\":null}")]
    [InlineData("[1, [2, 3], {\"a\": 4}]")]
    [InlineData("{\"a\": {}}")]
    [InlineData("{\"a\": \"\\\"\"}")]
    [InlineData("{\"a\": \"\\\\\"}")]
    [InlineData("{\"a\": \"\\/\"}")]
    [InlineData("{\"a\": \"\\b\"}")]
    [InlineData("{\"a\": \"\\f\"}")]
    [InlineData("{\"a\": \"\\n\"}")]
    [InlineData("{\"a\": \"\\r\"}")]
    [InlineData("{\"a\": \"\\t\"}")]
    [InlineData("{\"\\\"\":\"b\"}")]
    [InlineData("{\"\\\\\":\"b\"}")]
    [InlineData("{\"\\/\":\"b\"}")]
    [InlineData("{\"\\b\":\"b\"}")]
    [InlineData("{\"\\f\":\"b\"}")]
    [InlineData("{\"\\n\":\"b\"}")]
    [InlineData("{\"\\r\":\"b\"}")]
    [InlineData("{\"\\t\":\"b\"}")]
    public void TestValidJson(string json)
    {
        var sm = new JsonStateMachine();
        sm.Process(json);
        sm.Complete();
        Assert.True(sm.IsComplete);
    }

    [Theory]
    [InlineData("")]
    [InlineData("{\"key\":}")]
    [InlineData("[1,]")]
    [InlineData("[],")]
    [InlineData("[] ")]
    [InlineData("[1 2]")]
    [InlineData("{")]
    [InlineData("[")]
    [InlineData("{\"key\":\"value\",}")]
    [InlineData("abc")]
    [InlineData("1, 2, 3")]
    [InlineData("{\"my\nkey\":\"value\"}")]
    [InlineData("{\"key\":\"value\n\"}")]
    [InlineData("{\"a\": \"\\z\"}")]
    public void TestInvalidJson(string json)
    {
        var sm = new JsonStateMachine();
        sm.Process(json);
        sm.Complete();
        Assert.False(sm.IsValid);
    }

    [Theory]
    [InlineData("{\"key", JsonState.InStringKey)]
    [InlineData("{\"key\":\"value", JsonState.InString)]
    [InlineData("{", JsonState.ExpectingFirstKey)]
    [InlineData("[", JsonState.ExpectingFirstValue)]
    [InlineData("{\"key\":", JsonState.ExpectingValue)]
    [InlineData("{\"key\":\"value\",", JsonState.ExpectingKey)]
    [InlineData("[1", JsonState.InNumber)]
    [InlineData("[1,", JsonState.ExpectingValue)]
    [InlineData("[true,", JsonState.ExpectingValue)]
    public void TestIncompleteJson(string json, JsonState expected)
    {
        var sm = new JsonStateMachine();
        sm.Process(json);

        var transition = sm.Transition as JsonTransition;
        Assert.NotNull(transition);
        Assert.Equal(expected, transition.State);
    }

    [Fact]
    public void TestStateSequenceForSimpleObject()
    {
        var json = "{\"key\":\"value\"}";
        var states = new List<JsonState>();
        var sm = new JsonStateMachine();
        sm.StateChanged += states.Add;
        sm.Process(json);
        sm.Complete();

        Assert.True(sm.IsComplete);
        Assert.Equal(new[] {
            JsonState.ExpectingFirstKey,
            JsonState.InStringKey,
            JsonState.ExpectingColon,
            JsonState.ExpectingValue,
            JsonState.InString,
            JsonState.ExpectingCommaOrEnd,
            JsonState.Complete
        }, states);
    }

}
