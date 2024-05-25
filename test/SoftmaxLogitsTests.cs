namespace libLlama2.UnitTests;

using ManagedCuda;

public class SoftmaxLogitsTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public SoftmaxLogitsTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("softmax_logits_kernel.ptx", "softmax_logits_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private Half[] SoftmaxLogits(Half[] x, float temperature, out int[] indices)
    {
        var size = x.Length;
        var x1 = (CudaDeviceVariable<Half>)x;
        var x2 = new CudaDeviceVariable<int>(size);
        kernel.Run(x1.DevicePointer, size, temperature, x2.DevicePointer);
        indices = (int[])x2;
        return (Half[])x1;
    }

    private Half[] Expected(Half[] logits, float temperature, out int[] indices)
    {

        var size = logits.Length;
        indices = new int[size];

        for (var i = 0; i < size; ++i)
        {
            indices[i] = i;

            float val = (float)logits[i];
            val /= temperature;
            logits[i] = (Half)val;
        }

        // find max value (for numerical stability)
        float max_val = (float)logits[0];
        for (int i = 1; i < size; i++)
        {
            if ((float)logits[i] > max_val)
            {
                max_val = (float)logits[i];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            float v = MathF.Exp((float)logits[i] - max_val);
            logits[i] = (Half)v;
            sum += v;
        }
        // normalize
        for (int i = 0; i < size; i++)
        {
            logits[i] = (Half)((float)logits[i] / sum);
        }

        return logits;
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int count = 10;
        for (int i = 0; i < count; ++i)
            yield return new object[] { generator.NextArray(16), 1.0f };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_SoftmaxLogits(Half[] x, float temperature)
    {
        var actual = SoftmaxLogits(x, temperature, out var actualIndices);

        var expected = Expected(x, temperature, out var expectedIndices);

        Assert.Equal(expected, actual);
        Assert.Equal(expectedIndices, actualIndices);
    }
}
