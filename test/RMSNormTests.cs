namespace libLlama2.UnitTests;

using ManagedCuda;

public class RMSNormTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public RMSNormTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("rmsnorm_kernel.ptx", "rmsnorm_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    public Half[] RMSNorm(Half[] vector1, Half[] vector2)
    {
        var v1 = (CudaDeviceVariable<Half>)vector1;
        var v2 = (CudaDeviceVariable<Half>)vector2;
        var size = v1.Size;
        var result = new CudaDeviceVariable<Half>(size);
        var elementsPerThread = DivUp(size, 1024);
        kernel.Run(result.DevicePointer, v1.DevicePointer, v2.DevicePointer, size, elementsPerThread);

        return (Half[])result;
    }

    // Reference implementation
    private Half[] Expected(Half[] input, Half[] gamma)
    {
        var size = input.Length;
        var result = new float[size];
        var ss = 0.0f;
        for (int i = 0; i < size; ++i)
            ss += (float)(input[i] * input[i]);

        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / MathF.Sqrt(ss);
        for (int i = 0; i < size; ++i)
        {
            result[i] = (float)gamma[i] * (ss * (float)input[i]);
        }

        return ToHalf(result);
    }

    private int DivUp(int a, int b)
    {
        return (a - 1) / b + 1;
    }

    private Half[] ToHalf(float[] values)
    {
        return values.Select(x => (Half)x).ToArray();
    }

    [Theory]
    [InlineData(new[] { 1f, 2f, 3f }, new[] { 4f, 5f, 6f })]
    [InlineData(new[] { 6f, 2f, 6f }, new[] { 1f, 9f, 7f })]
    public void Test_RMSNorm(float[] vector1, float[] vector2)
    {
        var input = ToHalf(vector1);
        var weight = ToHalf(vector2);

        var actual = RMSNorm(input, weight);
        var expected = Expected(input, weight);

        Assert.Equal(expected, actual);
    }
}
