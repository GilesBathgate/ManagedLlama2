namespace libLlama2.UnitTests;

public class Generator
{
    private readonly Random random = new(31415);

    public Half[] NextArray(int size)
    {
        var values = new Half[size];
        for (int i = 0; i < size; ++i)
            values[i] = (Half)random.NextDouble();

        return values;
    }

    public int[] NextIntArray(int size, int min, int max)
    {
        var values = new int[size];
        for (int i = 0; i < size; ++i)
            values[i] = random.Next(min, max);

        return values;
    }
}
