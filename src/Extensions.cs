namespace libLlama2;

public static class Extensions
{
    public static float[] ToFloat(this Half[] values) =>
        values.Select(x => (float)x).ToArray();

    public static Half[] ToHalf(this float[] values) =>
        values.Select(x => (Half)x).ToArray();
}
