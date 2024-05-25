namespace libLlama2.UnitTests;

using ManagedCuda;

public class ArgmaxTests : IDisposable
{
    private readonly CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public ArgmaxTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("argmax_kernel.ptx", "argmax_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int Argmax(Half[] logits, int n)
    {
        var x = (CudaDeviceVariable<Half>)logits;
        var result = new CudaDeviceVariable<int>(1);

        var pPos = (CudaDeviceVariable<int>)(-1);
        kernel.Run(x.DevicePointer, n, result.DevicePointer, pPos.DevicePointer, pPos.DevicePointer, true);

        return result[0];
    }

    private static int Expected(Half[] logits, int n)
    {
        var max_i = 0;
        Half max_p = logits[0];
        for (var i = 1; i < n; ++i)
        {
            if (logits[i] > max_p)
            {
                max_i = i;
                max_p = logits[i];
            }
        }
        return max_i;
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int size = 10;
        yield return new object[] { generator.NextArray(size), size };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_Softmax(Half[] logits, int size)
    {
        var actual = Argmax(logits, size);

        var expected = Expected(logits, size);

        Assert.Equal(expected, actual);
    }
}
