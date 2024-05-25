namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class MatVecTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public MatVecTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] MatVec(Half[] matrix, Half[] vector, int rows, int cols)
    {
        var m = (CudaDeviceVariable<Half>)matrix;
        var v = (CudaDeviceVariable<Half>)vector;
        var result = new CudaDeviceVariable<Half>(v.Size);

        int serialLoads = ceil_div(ceil_div(rows, 32), 8);

        kernel.GridDimensions = new dim3(ceil_div(cols, 4), 1);
        kernel.BlockDimensions = new dim3(32, 32);
        kernel.DynamicSharedMemory = 0;

        kernel.Run(result.DevicePointer, v.DevicePointer, m.DevicePointer, rows, cols, serialLoads, 0, 0, 0, rows, 1.0f);
        return (Half[])result;
    }

    private Half[] Expected(Half[] w, Half[] x, int n, int d)
    {
        var xout = new float[n];

        for (int i = 0; i < d; ++i)
        {
            float val = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                val += (float)w[i * n + j] * (float)x[j];
            }
            xout[i] = val;
        }

        return xout.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int rows = 4096;
        const int cols = 64;
        yield return new object[] { generator.NextArray(rows * cols), generator.NextArray(rows), rows, cols };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_MatVec(Half[] w, Half[] x, int rows, int cols)
    {
        var expected = Expected(w, x, rows, cols);

        var actual = MatVec(w, x, rows, cols);

        Assert.Equal(expected, actual);
    }
}
