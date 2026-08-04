using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace libLlama2;

public class DictionaryStateMachine : IConstraintStateMachine, IDynamicConstraintStateMachine
{
    private readonly HashSet<string> words;
    private readonly TrieNode root;
    private string currentPrefix = string.Empty;

    public ITransition Transition => new DictionaryTransition(currentPrefix);

    public DictionaryStateMachine(string filePath)
    {
        words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        root = new TrieNode();

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Dictionary file not found: {filePath}");
        }

        foreach (var line in File.ReadLines(filePath))
        {
            var word = line.Trim().ToLowerInvariant();
            if (!string.IsNullOrEmpty(word))
            {
                words.Add(word);
                InsertIntoTrie(word);
            }
        }
    }

    public DictionaryStateMachine(IEnumerable<string> wordList)
    {
        words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        root = new TrieNode();

        foreach (var w in wordList)
        {
            var word = w.Trim().ToLowerInvariant();
            if (!string.IsNullOrEmpty(word))
            {
                words.Add(word);
                InsertIntoTrie(word);
            }
        }
    }

    private void InsertIntoTrie(string word)
    {
        var current = root;
        foreach (var c in word)
        {
            if (!current.Children.TryGetValue(c, out var next))
            {
                next = new TrieNode();
                current.Children[c] = next;
            }
            current = next;
        }
    }

    private bool IsPrefixOfAnyWord(string prefix)
    {
        var current = root;
        foreach (var c in prefix)
        {
            if (!current.Children.TryGetValue(c, out current))
            {
                return false;
            }
        }
        return true;
    }

    public IEnumerable<ITransition> PossibleTransitions()
    {
        return new[] { Transition };
    }

    public void Process(Token token)
    {
        currentPrefix = GetNewPrefix(currentPrefix, token.Value);
    }

    public void Reset()
    {
        currentPrefix = string.Empty;
    }

    public bool IsComplete => false;

    public (bool allowed, List<int> tokenIds) GetActiveTokens(ITokenizer tokenizer, int vocabSize)
    {
        var validTokenIds = new List<int>();
        var invalidTokenIds = new List<int>();

        for (int i = 0; i < vocabSize; i++)
        {
            var tokenText = tokenizer.Decode(i);
            if (IsValidContinuation(currentPrefix, tokenText))
            {
                validTokenIds.Add(i);
            }
            else
            {
                invalidTokenIds.Add(i);
            }
        }

        if (validTokenIds.Count <= invalidTokenIds.Count)
        {
            return (true, validTokenIds);
        }
        else
        {
            return (false, invalidTokenIds);
        }
    }

    private static bool IsWordChar(char c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');

    private static bool ContainsWordChar(string s)
    {
        foreach (char c in s)
        {
            if (IsWordChar(c)) return true;
        }
        return false;
    }

    private string GetNewPrefix(string prefix, string tokenText)
    {
        string full = prefix + tokenText;
        int i = full.Length - 1;
        while (i >= 0 && IsWordChar(full[i]))
        {
            i--;
        }
        return full.Substring(i + 1);
    }

    private bool IsValidContinuation(string prefix, string tokenText)
    {
        string full = prefix + tokenText;
        int i = 0;
        while (i < full.Length)
        {
            if (IsWordChar(full[i]))
            {
                int start = i;
                while (i < full.Length && IsWordChar(full[i]))
                {
                    if (i > start && char.IsUpper(full[i]))
                    {
                        return false;
                    }
                    i++;
                }
                string word = full.Substring(start, i - start).ToLowerInvariant();

                if (i == full.Length)
                {
                    if (!IsPrefixOfAnyWord(word))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!words.Contains(word))
                    {
                        return false;
                    }
                }
            }
            else
            {
                i++;
            }
        }

        if (!string.IsNullOrEmpty(prefix) && (tokenText.Length == 0 || !ContainsWordChar(tokenText)))
        {
            if (!words.Contains(prefix.ToLowerInvariant()))
            {
                return false;
            }
        }

        return true;
    }

    private class TrieNode
    {
        public Dictionary<char, TrieNode> Children { get; } = new();
    }
}

public class DictionaryTransition : ITransition
{
    public string Prefix { get; }

    public DictionaryTransition(string prefix)
    {
        Prefix = prefix;
    }

    public bool CanBeFollowedBy(Token token)
    {
        return true;
    }

    public override bool Equals(object? obj) =>
        obj is DictionaryTransition other && Prefix == other.Prefix;

    public override int GetHashCode() =>
        Prefix.GetHashCode();
}
