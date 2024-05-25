namespace libLlama2.UnitTests;

using ManagedCuda;

public class SampleTopPTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public SampleTopPTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("sample_top_p_kernel.ptx", "sample_top_p_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int SampleTopP(Half[] x, int[] indices, float threshold)
    {
        var size = x.Length;
        var x1 = (CudaDeviceVariable<Half>)x;
        var x2 = (CudaDeviceVariable<int>)indices;
        var result = new CudaDeviceVariable<int>(1);
        kernel.Run(x1.DevicePointer, x2.DevicePointer, size, threshold, result.DevicePointer, 0);

        return result[0];
    }

    private int Expected(Half[] logits, int[] indices, float threshold)
    {
        var n = indices.Length;

        int min_index = n - 1;
        for (int i = 0; i < n; i++)
        {
            if ((float)logits[i] >= threshold && i < min_index)
                min_index = i;
        }

        return indices[min_index];
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int count = 10;
        for (int i = 0; i < count; ++i)
            yield return new object[] { generator.NextArray(16), generator.NextIntArray(16, 0, 16).ToArray(), 0.9f };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_SampleTopP(Half[] x, int[] indices, float threshold)
    {
        var actual = SampleTopP(x, indices, threshold);

        var expected = Expected(x, indices, threshold);

        Assert.Equal(expected, actual);
    }
}
