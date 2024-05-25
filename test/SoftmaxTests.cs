namespace libLlama2.UnitTests;

using ManagedCuda;

public class SoftmaxTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public SoftmaxTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("softmax_kernel.ptx", "softmax_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private Half[] Softmax(Half[] x)
    {
        var size = x.Length;
        var x1 = (CudaDeviceVariable<Half>)x;
        kernel.Run(x1.DevicePointer, size);
        return (Half[])x1;
    }

    private Half[] Expected(Half[] x1)
    {
        var x = x1.ToFloat();

        int size = x.Length;
        // find max value (for numerical stability)
        float max_val = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > max_val)
            {
                max_val = x[i];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[i] = MathF.Exp(x[i] - max_val);
            sum += x[i];
        }
        // normalize
        for (int i = 0; i < size; i++)
        {
            x[i] /= sum;
        }

        return x.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int count = 10;
        for (int i = 0; i < count; ++i)
            yield return new[] { generator.NextArray(16) };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_Softmax(Half[] x)
    {
        var actual = Softmax(x);

        var expected = Expected(x);

        Assert.Equal(expected, actual);
    }
}
