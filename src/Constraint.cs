namespace libLlama2;

public class Constraint
{
    public bool Allowed { get; }

    public int Index { get; }

    public int Size { get; }

    public Constraint(bool allowed, int index, int size)
    {
        Allowed = allowed;
        Index = index;
        Size = size;
    }
}