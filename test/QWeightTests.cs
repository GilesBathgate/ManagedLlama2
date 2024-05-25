namespace libLlama2.UnitTests;

public class QWeightTests
{

    public static T[] Expand<T>(T value, int count = 8) =>
        Enumerable.Repeat(value, count).ToArray();

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        yield return new object[] { generator.NextIntArray(1024, 0, 15).ToHalf(), Expand(1.0f), Expand(0) };
        yield return new object[] { generator.NextIntArray(1024, -1, 14).ToHalf(), Expand(1.0f), Expand(1) };
        yield return new object[] { generator.NextIntArray(1024, -14, 1).ToHalf(), Expand(1.0f), Expand(14) };
        yield return new object[] { generator.NextIntArray(1024, -7, 8).ToHalf(), Expand(1.0f), Expand(7) };
        yield return new object[] { generator.NextIntArray(1024, 0, 15).Select(x => x * 10).ToHalf(), Expand(10f), Expand(0) };
        yield return new object[] { generator.NextIntArray(1024, 0, 15).Select(x => x / 10).ToHalf(), Expand(0.1f), Expand(0) };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_QWeight_Pack_Unpack(Half[] expected, float[] scales, uint[] zeros)
    {
        var actual = HostQWeight.Pack(expected, scales, zeros).Unpack();

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Test_QWeight_FromData()
    {
        var data = new QWeightTestData();

        uint combined = 0;
        for (var z = 0; z < 8; ++z)
            combined |= (data.zeros[z] & 0xF) << (z * 4);

        Assert.Equal(data.zero, combined);

        var packed = new HostQWeight(128);
        packed.Zeros[0] = combined;
        packed.Scales = data.scales.ToHalf();
        packed.Weight = data.weights;

        var x = packed.Unpack();

        Assert.Equal(x, data.checks.ToHalf());
    }
}
