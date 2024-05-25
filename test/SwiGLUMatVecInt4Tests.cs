namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class SwiGLUMatVecInt4Tests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public SwiGLUMatVecInt4Tests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_swiglu_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] SwiGLUMatVecInt4(HostQWeight g, HostQWeight u, Half[] x, int rows, int cols)
    {
        var scales_height = ceil_div(rows, 128);
        var packed_wt_height = ceil_div(rows, 32) * 4;
        int packed_zeros_height = ceil_div(scales_height, 8);
        kernel.BlockDimensions = new dim3(32, 4);
        kernel.GridDimensions = new dim3(ceil_div(cols, 4), 1);

        var gwt = (CudaDeviceVariable<uint>)g.Weight;
        var gze = (CudaDeviceVariable<uint>)g.Zeros;
        var gsc = (CudaDeviceVariable<Half>)g.Scales;

        var uwt = (CudaDeviceVariable<uint>)u.Weight;
        var uze = (CudaDeviceVariable<uint>)u.Zeros;
        var usc = (CudaDeviceVariable<Half>)u.Scales;

        var xi = (CudaDeviceVariable<Half>)x;

        var xout = new CudaDeviceVariable<Half>(x.Length);

        kernel.Run(xout.DevicePointer, xi.DevicePointer,
                   gwt.DevicePointer, gze.DevicePointer, gsc.DevicePointer,
                   uwt.DevicePointer, uze.DevicePointer, usc.DevicePointer,
                   rows, cols, packed_zeros_height, scales_height, packed_wt_height);

        return (Half[])xout;
    }

    private static float[] MatVec(Half[] w, Half[] x, int n, int d)
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

        return xout;
    }

    private static Half[] Expected(Half[] gw, Half[] uw, Half[] x, int n, int d)
    {

        var g_val = MatVec(gw, x, n, d);
        var u_val = MatVec(uw, x, n, d);

        var hb = new float[n];

        // SwiGLU non-linearity
        for (int i = 0; i < d; ++i)
        {
            float val = g_val[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= 1.0f / (1.0f + MathF.Exp(-val));
            // elementwise multiply with w3(x)
            val *= u_val[i];
            hb[i] = val;
        }

        return hb.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int rows = 128;
        const int cols = 128;
        const float scale = 0.01f;
        var scales = Enumerable.Repeat(scale, 128).ToArray();
        var zeros = Enumerable.Repeat(0u, 128).ToArray();
        yield return new object[] { generator.NextIntArray(rows * cols, 0, 15).Select(x => x * scale).ToHalf(),
                                    generator.NextIntArray(rows * cols, -15, 0).Select(x => x * scale).ToHalf(),
                                    generator.NextArray(cols),
                                    scales, zeros, rows, cols };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_SwiGLUMatVecInt4(Half[] g, Half[] u, Half[] x, float[] scales, uint[] zeros, int rows, int cols)
    {
        var packed_g = HostQWeight.Pack(g, scales, zeros);
        var packed_u = HostQWeight.Pack(u, scales, zeros);

        var expected = Expected(packed_g.Unpack(), packed_u.Unpack(), x, rows, cols);

        var actual = SwiGLUMatVecInt4(packed_g, packed_u, x, rows, cols);

        var expectedApprox = expected.Select(Half.Truncate);
        var actualApprox = actual.Select(Half.Truncate);
        Assert.Equal(expectedApprox, actualApprox);
    }
}
