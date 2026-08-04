namespace libLlama2;

public interface ITransition
{
    bool CanBeFollowedBy(Token token);
    bool Equals(object? obj);
    int GetHashCode();
}