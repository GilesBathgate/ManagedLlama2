using Moq;

using JsonState = libLlama2.JsonStateMachine.JsonState;

namespace libLlama2.UnitTests;

public class JsonStateMachineTests
{
    private readonly Mock<ITokenizer> tokenizerMock;

    public JsonStateMachineTests()
    {
        tokenizerMock = new Mock<ITokenizer>();
    }

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
        Assert.Equal(JsonState.Complete, sm.State);
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
        Assert.Equal(JsonState.Error, sm.State);
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
        Assert.Equal(expected, sm.State);
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

        Assert.Equal(JsonState.Complete, sm.State);
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

    public static IEnumerable<object[]> PrecomputationOverloadTestData()
    {
        var initialObjectContext = JsonState.InObject;
        yield return new object[] {
            initialObjectContext, JsonState.ExpectingKey,
            new[] { new Token(0, "\"key\"") },
            new[] { new Token(1, "}"), new Token(2, ":") }
        };

        yield return new object[] {
            JsonState.Initial, JsonState.ExpectingFirstValue,
            new[] { new Token(0, "{") , new Token(1, "["), new Token(2, "true") },
            new[] { new Token(3, "F") }
        };

        yield return new object[] {
            initialObjectContext, JsonState.ExpectingCommaOrEnd,
            new[] { new Token(0, "}"), new Token(1, ",") },
            new[] { new Token(2, "]") }
        };

        var initialArrayContext = JsonState.InArray;
        yield return new object[] {
            initialArrayContext, JsonState.ExpectingValue,
            new[] { new Token(0, "123"), new Token(1, "true"), new Token(2, "\"hello\""), new Token(3, "{"), new Token(4, "[") },
            new[] { new Token(5, ","), new Token(6, ":") }
        };

        yield return new object[] {
            initialArrayContext, JsonState.ExpectingCommaOrEnd,
            new[] { new Token(0, "]"), new Token(1, ",") },
            new[] { new Token(2, "}") }
        };
    }

    [Theory]
    [MemberData(nameof(PrecomputationOverloadTestData))]
    public void TestPrecomputation_WithFixedContext(JsonState context, JsonState state, Token[] allowedTokens, Token[] disallowedTokens)
    {
        var sm = new JsonStateMachine();
        var allTokens = allowedTokens.Concat(disallowedTokens).ToList();
        var precomputed = It.IsAny<Dictionary<(JsonState, JsonState), ConstraintGenerator.Constraint>>();
        var generator = new ConstraintGenerator(tokenizerMock.Object, allTokens.Count, precomputed);

        foreach (var token in allTokens)
            tokenizerMock.Setup(t => t.Decode(token.Id)).Returns(token.Value);

        var validTokens = generator.PrecomputeValidTokens(context);
        var computedAllowedTokens = validTokens[state];

        foreach (var token in allowedTokens)
            Assert.Contains(token, computedAllowedTokens);

        foreach (var token in disallowedTokens)
            Assert.DoesNotContain(token, computedAllowedTokens);
    }


    [Fact]
    public void TestLlmUsageSimulation()
    {
        // 1. Define a vocabulary and set up the tokenizer and state machine
        var keyToken = new Token(0, "\"key\"");
        var colonToken = new Token(1, ":");
        var trueToken = new Token(2, "true");
        var falseToken = new Token(3, "false");
        var openBraceToken = new Token(4, "{");
        var closeBraceToken = new Token(5, "}");

        var allTokens = new[] { keyToken, colonToken, trueToken, falseToken, openBraceToken, closeBraceToken };
        foreach (var token in allTokens)
            tokenizerMock.Setup(t => t.Decode(token.Id)).Returns(token.Value);

        var precomputed = It.IsAny<Dictionary<(JsonState, JsonState), ConstraintGenerator.Constraint>>();
        var generator = new ConstraintGenerator(tokenizerMock.Object, allTokens.Length, precomputed);
        var sm = new JsonStateMachine();

        // 2. Precompute the lookup table of valid next tokens
        var validTokensLookup = generator.PrecomputeValidTokens();

        // 3. Start of generation: LLM is prompted to produce a JSON object. It emits "{"
        sm.Process("{");

        // 4. Verify the new state and get the list of valid next tokens
        Assert.Equal(JsonState.ExpectingFirstKey, sm.State);
        Assert.Equal(JsonState.InObject, sm.CurrentContext);
        var nextTokens = validTokensLookup[(sm.State, sm.CurrentContext)];

        // A key or a closing brace are valid next tokens
        Assert.Contains(keyToken, nextTokens);
        Assert.Contains(closeBraceToken, nextTokens);
        Assert.DoesNotContain(colonToken, nextTokens); // A colon is not valid here

        // 5. LLM selects a valid token: "\"key\""
        sm.Process(keyToken.Value);

        // 6. Verify the new state and get the next constraints
        Assert.Equal(JsonState.ExpectingColon, sm.State);
        Assert.Equal(JsonState.InObject, sm.CurrentContext);
        nextTokens = validTokensLookup[(sm.State, sm.CurrentContext)];

        // Only a colon is valid
        Assert.Contains(colonToken, nextTokens);
        Assert.DoesNotContain(trueToken, nextTokens);

        // 7. LLM selects a valid token: ":"
        sm.Process(colonToken.Value);

        // 8. Verify new state and constraints
        Assert.Equal(JsonState.ExpectingValue, sm.State);
        Assert.Equal(JsonState.InObject, sm.CurrentContext);
        nextTokens = validTokensLookup[(sm.State, sm.CurrentContext)];

        // A value (like true/false) is valid. A string is also a valid value.
        Assert.Contains(trueToken, nextTokens);
        Assert.Contains(falseToken, nextTokens);
        Assert.Contains(keyToken, nextTokens); // A string like "key" is a valid value
        Assert.DoesNotContain(colonToken, nextTokens); // A colon is not a valid value

        // 9. LLM selects "true"
        sm.Process(trueToken.Value);

        // 10. Verify new state and constraints
        Assert.Equal(JsonState.ExpectingCommaOrEnd, sm.State);
        Assert.Equal(JsonState.InObject, sm.CurrentContext);
        nextTokens = validTokensLookup[(sm.State, sm.CurrentContext)];

        // A closing brace is valid
        Assert.Contains(closeBraceToken, nextTokens);
        Assert.DoesNotContain(keyToken, nextTokens);

        // 11. LLM selects "}"
        sm.Process(closeBraceToken.Value);

        // 12. Final state should be Complete
        Assert.Equal(JsonState.Complete, sm.State);
        Assert.Equal(default, sm.CurrentContext);
    }

    [Fact]
    public void TestPrecomputation_ConstrainedTokens_UsesDisallowedSet_WhenSmaller()
    {
        // Arrange
        var invalidToken = new Token(0, "\n"); // Newline is invalid in a string literal
        var validToken1 = new Token(1, "a");
        var validToken2 = new Token(2, "b");
        var validToken3 = new Token(3, "c");
        var validToken4 = new Token(4, "\""); // Quote is valid, it ends the string

        var allTokens = new[] { invalidToken, validToken1, validToken2, validToken3, validToken4 };
        foreach (var token in allTokens)
            tokenizerMock.Setup(t => t.Decode(token.Id)).Returns(token.Value);

        // We are in a string, inside an array context
        var precomputed = It.IsAny<Dictionary<(JsonState, JsonState), ConstraintGenerator.Constraint>>();
        var generator = new ConstraintGenerator(tokenizerMock.Object, allTokens.Length, precomputed);

        // Use reflection to get the precomputed tokens
        var precomputedConstrains = generator.PrecomputeConstraints();

        var key = (JsonState.InString, JsonState.InArray);
        var constraint = precomputedConstrains[key];

        // Assert
        // The only invalid token is \n. The other 4 are valid.
        // The new implementation should store the 1 invalid token.
        Assert.False(constraint.Allowed); // It should store the disallowed set
        Assert.Equal(1, constraint.Tokens.Count);
        Assert.Contains(invalidToken, constraint.Tokens);
    }
}
