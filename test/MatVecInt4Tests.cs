namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class MatVecInt4Tests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public MatVecInt4Tests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_residual_int4_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] MatVecInt4(HostQWeight w, Half[] x, int rows, int cols)
    {
        var inpSize = cols;
        var opSize = cols;
        var scales_height = ceil_div(inpSize, 128);
        var packed_wt_height = ceil_div(inpSize, 32) * 4;
        int packed_zeros_height = ceil_div(scales_height, 8);
        kernel.BlockDimensions = new dim3(32, 4);
        kernel.GridDimensions = new dim3(ceil_div(opSize, 4), 1);

        var wt = (CudaDeviceVariable<uint>)w.Weight;
        var ze = (CudaDeviceVariable<uint>)w.Zeros;
        var sc = (CudaDeviceVariable<Half>)w.Scales;
        var xout = (CudaDeviceVariable<Half>)x;

        kernel.Run(xout.DevicePointer, xout.DevicePointer, wt.DevicePointer, ze.DevicePointer, sc.DevicePointer,
                   inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height);

        return (Half[])xout;
    }

    private Half[] Expected(Half[] w, Half[] x, int n, int d)
    {
        var xout = x.ToFloat();

        var xb2 = new float[n];
        for (int i = 0; i < d; ++i)
        {
            float val = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                val += (float)w[i * n + j] * (float)x[j];
            }
            xb2[i] = val;
        }

        // residual connection back into x
        for (int i = 0; i < d; i++)
        {
            xout[i] += xb2[i];
        }

        return xout.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int rows = 128;
        const int cols = 128;
        var scales = Enumerable.Repeat(1f, 128).ToArray();
        var zeros = Enumerable.Repeat(0u, 128).ToArray();
        yield return new object[] { generator.NextIntArray(rows * cols, 0, 15).ToHalf(), generator.NextArray(cols), scales, zeros, rows, cols };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_MatVecInt4(Half[] w, Half[] x, float[] scales, uint[] zeros, int rows, int cols)
    {
        var q = HostQWeight.Pack(w, scales, zeros);

        var expected = Expected(q.Unpack(), x, rows, cols);

        var actual = MatVecInt4(q, x, rows, cols);

        Assert.Equal(expected, actual);
    }
}
