namespace libLlama2.UnitTests;

using ManagedCuda;

public class CumulativeSumTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public CumulativeSumTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("cumulative_sum_kernel.ptx", "cumulative_sum_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] CumulativeSum(Half[] x)
    {
        var size = x.Length;
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;

        var x1 = (CudaDeviceVariable<Half>)x;
        kernel.Run(x1.DevicePointer, size);

        return (Half[])x1;
    }

    private Half[] Expected(Half[] probabilities)
    {
        var size = probabilities.Length;
        var output = new float[size];
        var cdf = 0.0f;
        for (int i = 0; i < size; i++)
        {
            cdf += (float)probabilities[i];
            output[i] = cdf;
        }

        return output.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        var vocabSize = 32_000;
        for (int i = 0; i < 10; ++i)
            yield return new[] { generator.NextIntArray(vocabSize, -1000, 1001).ToHalf() };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_CumulativeSum(Half[] x)
    {
        var actual = CumulativeSum(x);

        var expected = Expected(x);

        Assert.Equal(expected, actual);
    }
}
