namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.BasicTypes;

public class VecAddTests
{
    private CudaKernel kernel;

    public VecAddTests()
    {
        const string kernelPath = "vecadd_kernel.ptx";
        const string kernelName = "VecAdd";

        int deviceID = 0;
        var ctx = new PrimaryContext(deviceID);
        ctx.SetCurrent();

        kernel = ctx.LoadKernel(kernelPath, kernelName);
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 256;
    }

    public float[] VecAdd(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length) throw new ArgumentException("Vectors are different sizes");

        var v1 = (CudaDeviceVariable<float>)vector1;
        var v2 = (CudaDeviceVariable<float>)vector2;

        var size = v1.Size;
        var result = new CudaDeviceVariable<float>(size);
        kernel.Run(v1.DevicePointer, v2.DevicePointer, result.DevicePointer, size);
        return (float[])result;
    }

    [Theory]
    [InlineData(new[] { 1f, 2f, 3f }, new[] { 4f, 5f, 6f }, new[] { 5f, 7f, 9f })]
    [InlineData(new[] { 1f, -2f, 3f }, new[] { -4f, 5f, -6f }, new[] { -3f, 3f, -3f })]
    public void Test_VecAdd_Basic(float[] vector1, float[] vector2, float[] expected)
    {
        var actual = VecAdd(vector1, vector2);
        Assert.Equal(expected, actual);
    }

}
