namespace libLlama2.UnitTests;

using ManagedCuda;

public class ConvertTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public ConvertTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("convert_kernel.ptx", "convert_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    public float[] Convert(Half[] vector1)
    {
        var v1 = (CudaDeviceVariable<Half>)vector1;
        var size = v1.Size;
        var result = new CudaDeviceVariable<float>(size);
        kernel.Run(result.DevicePointer, v1.DevicePointer, size);
        return (float[])result;
    }

    private Half[] ToHalf(float[] values)
    {
        return values.Select(x => (Half)x).ToArray();
    }

    [Theory]
    [InlineData(new[] { 1f, 2f, 3f })]
    [InlineData(new[] { 6f, 2f, 6f })]
    public void Test_Convert(float[] expected)
    {
        var input = ToHalf(expected);

        var actual = Convert(input);

        Assert.Equal(expected, actual);
    }
}
