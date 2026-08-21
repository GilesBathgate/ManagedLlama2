using Moq;
using Xunit;
using System.Collections.Generic;
using libLlama2;

namespace libLlama2.UnitTests;

public class DictionaryStateMachineTests
{
    private readonly string[] sampleWords = new[] { "the", "there", "apple", "pie", "good" };

    [Fact]
    public void TestValidWordPrefixesAndWords()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        // Initially prefix is empty, which is valid.
        Assert.Equal(string.Empty, sm.Transition.GetType().GetProperty("Prefix")?.GetValue(sm.Transition));

        // Process "app"
        sm.Process(new Token(1, "app"));
        // Now prefix is "app", which is a prefix of "apple"
        Assert.True(sm.Transition.Equals(new DictionaryTransition("app")));

        // Process "le"
        sm.Process(new Token(2, "le"));
        // Now prefix is "apple", which is a complete word
        Assert.True(sm.Transition.Equals(new DictionaryTransition("apple")));

        // Process " "
        sm.Process(new Token(3, " "));
        // Space should reset currentPrefix because the word ended
        Assert.True(sm.Transition.Equals(new DictionaryTransition(string.Empty)));
    }

    [Fact]
    public void TestInvalidWordEnding()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        // "app" is not a complete word, so checking if space can follow it should fail
        var tokenizerMock = new Mock<ITokenizer>();
        tokenizerMock.Setup(t => t.Decode(0)).Returns(" "); // space
        tokenizerMock.Setup(t => t.Decode(1)).Returns("le"); // valid continuation

        sm.Process(new Token(1, "app"));

        // Let's check GetActiveTokens
        var (allowed, tokenIds) = sm.GetActiveTokens(tokenizerMock.Object, 2);

        // "le" should be allowed, but " " should not be!
        var allowedSet = new HashSet<int>();
        if (allowed)
        {
            allowedSet.UnionWith(tokenIds);
        }
        else
        {
            allowedSet.UnionWith(new[] { 0, 1 });
            allowedSet.ExceptWith(tokenIds);
        }

        Assert.Contains(1, allowedSet); // "le" is valid
        Assert.DoesNotContain(0, allowedSet); // " " is invalid because "app" is not a complete word
    }

    [Fact]
    public void TestValidWordEnding()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        // "apple" is a complete word, so checking if space can follow it should succeed
        var tokenizerMock = new Mock<ITokenizer>();
        tokenizerMock.Setup(t => t.Decode(0)).Returns(" "); // space
        tokenizerMock.Setup(t => t.Decode(1)).Returns(" pie"); // valid space + word

        sm.Process(new Token(1, "apple"));

        var (allowed, tokenIds) = sm.GetActiveTokens(tokenizerMock.Object, 2);

        var allowedSet = new HashSet<int>();
        if (allowed)
        {
            allowedSet.UnionWith(tokenIds);
        }
        else
        {
            allowedSet.UnionWith(new[] { 0, 1 });
            allowedSet.ExceptWith(tokenIds);
        }

        Assert.Contains(0, allowedSet); // " " is valid because "apple" is a complete word
        Assert.Contains(1, allowedSet); // " pie" is valid because it ends "apple", then has "pie"
    }

    [Fact]
    public void TestMultiWordToken()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        var tokenizerMock = new Mock<ITokenizer>();
        // token is " pie is" -> "apple pie is" -> "is" is not in dict, so it should be invalid!
        tokenizerMock.Setup(t => t.Decode(0)).Returns(" pie is");
        // token is " pie good" -> "apple pie good" -> valid!
        tokenizerMock.Setup(t => t.Decode(1)).Returns(" pie good");

        sm.Process(new Token(1, "apple"));

        var (allowed, tokenIds) = sm.GetActiveTokens(tokenizerMock.Object, 2);

        var allowedSet = new HashSet<int>();
        if (allowed)
        {
            allowedSet.UnionWith(tokenIds);
        }
        else
        {
            allowedSet.UnionWith(new[] { 0, 1 });
            allowedSet.ExceptWith(tokenIds);
        }

        Assert.DoesNotContain(0, allowedSet); // " pie is" is invalid
        Assert.Contains(1, allowedSet); // " pie good" is valid
    }

    [Fact]
    public void TestCapitalizationConstraints()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        var tokenizerMock = new Mock<ITokenizer>();
        tokenizerMock.Setup(t => t.Decode(0)).Returns("apple"); // valid lowercase
        tokenizerMock.Setup(t => t.Decode(1)).Returns("Apple"); // valid title case
        tokenizerMock.Setup(t => t.Decode(2)).Returns("appLe"); // invalid mid-word uppercase
        tokenizerMock.Setup(t => t.Decode(3)).Returns("APPLE"); // invalid mid-word uppercase

        var (allowed, tokenIds) = sm.GetActiveTokens(tokenizerMock.Object, 4);

        var allowedSet = new HashSet<int>();
        if (allowed)
        {
            allowedSet.UnionWith(tokenIds);
        }
        else
        {
            allowedSet.UnionWith(new[] { 0, 1, 2, 3 });
            allowedSet.ExceptWith(tokenIds);
        }

        Assert.Contains(0, allowedSet); // "apple" (valid)
        Assert.Contains(1, allowedSet); // "Apple" (valid)
        Assert.DoesNotContain(2, allowedSet); // "appLe" (invalid)
        Assert.DoesNotContain(3, allowedSet); // "APPLE" (invalid)
    }

    [Fact]
    public void TestPunctuationConstraints()
    {
        var sm = new DictionaryStateMachine(sampleWords);

        var tokenizerMock = new Mock<ITokenizer>();
        tokenizerMock.Setup(t => t.Decode(0)).Returns("apple, pie"); // valid punctuation with space
        tokenizerMock.Setup(t => t.Decode(1)).Returns("apple."); // valid end of sentence
        tokenizerMock.Setup(t => t.Decode(2)).Returns("apple-pie"); // invalid mid-word hyphen
        tokenizerMock.Setup(t => t.Decode(3)).Returns("apple,pie"); // invalid mid-word comma without space

        var (allowed, tokenIds) = sm.GetActiveTokens(tokenizerMock.Object, 4);

        var allowedSet = new HashSet<int>();
        if (allowed)
        {
            allowedSet.UnionWith(tokenIds);
        }
        else
        {
            allowedSet.UnionWith(new[] { 0, 1, 2, 3 });
            allowedSet.ExceptWith(tokenIds);
        }

        Assert.Contains(0, allowedSet); // "apple, pie" (valid)
        Assert.Contains(1, allowedSet); // "apple." (valid)
        Assert.DoesNotContain(2, allowedSet); // "apple-pie" (invalid)
        Assert.DoesNotContain(3, allowedSet); // "apple,pie" (invalid)
    }
}
