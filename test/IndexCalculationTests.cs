namespace libLlama2.UnitTests;

public class IndexCalculationTests
{
    private static IEnumerable<(int, int)> SingleLoop(int size)
    {
        for (var x = 0; x < size; x++)
        {
            var y = 4 * (x / 4);
            yield return (x, y);
        }
    }

    private static IEnumerable<(int, int)> NestedLoop(int size)
    {
        for (var x = 0; x < size / 4; x++)
        {
            for (var y = 0; y < 4; y++)
            {
                var yIndex = x * 4;
                var xIndex = yIndex + y;
                yield return (xIndex, yIndex);
            }
        }
    }

    [Fact]
    public void Test_Indexing()
    {
        var expected = Enumerable.Range(0, 1024);
        var indices = new List<int>();

        const int xSize = 32;
        const int wSize = 4;
        const int iSize = 8;
        for (var x = 0; x < xSize; ++x)
        {
            for (var w = 0; w < wSize; ++w)
            {
                for (var i = 0; i < iSize; ++i)
                {
                    indices.Add(x * wSize * iSize + w * iSize + i);
                }
            }
        }

        Assert.Equal(expected, indices);
    }

    [Theory]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    public void CalculateIndex_BothApproaches_ProduceSameResult(int size)
    {
        var expected = Enumerable.Range(0, size).Select(x => (x, 4 * (x / 4)));

        Assert.Equal(expected, SingleLoop(size));
        Assert.Equal(expected, NestedLoop(size));
    }
}