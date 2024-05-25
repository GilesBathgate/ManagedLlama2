namespace libLlama2;

public static class Extensions
{
    public static float[] ToFloat(this IEnumerable<Half> values) =>
        values.Select(x => (float)x).ToArray();

    public static Half[] ToHalf(this IEnumerable<float> values) =>
        values.Select(x => (Half)x).ToArray();

    public static Half[] ToHalf(this IEnumerable<int> values) =>
        values.Select(x => (Half)x).ToArray();
}
