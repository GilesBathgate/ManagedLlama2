namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class QKVMatVecInt4Tests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public QKVMatVecInt4Tests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "qkv_matvec_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private (Half[] q, Half[] k, Half[] v) QKVMatVecInt4(Half[] xb, HostQWeight qw, HostQWeight kw, HostQWeight vw, int rows, int cols)
    {
        var inpSize = cols;
        var opSize = cols;
        var scales_height = ceil_div(inpSize, 128);
        var packed_wt_height = ceil_div(inpSize, 32) * 4;
        int packed_zeros_height = ceil_div(scales_height, 8);
        kernel.BlockDimensions = new dim3(32, 4);
        kernel.GridDimensions = new dim3(ceil_div(opSize, 4), 3);

        var xin = (CudaDeviceVariable<Half>)xb;

        var qwt = (CudaDeviceVariable<uint>)qw.Weight;
        var qze = (CudaDeviceVariable<uint>)qw.Zeros;
        var qsc = (CudaDeviceVariable<Half>)qw.Scales;

        var kwt = (CudaDeviceVariable<uint>)kw.Weight;
        var kze = (CudaDeviceVariable<uint>)kw.Zeros;
        var ksc = (CudaDeviceVariable<Half>)kw.Scales;

        var vwt = (CudaDeviceVariable<uint>)vw.Weight;
        var vze = (CudaDeviceVariable<uint>)vw.Zeros;
        var vsc = (CudaDeviceVariable<Half>)vw.Scales;

        var qout = new CudaDeviceVariable<Half>(cols);
        var kout = new CudaDeviceVariable<Half>(cols);
        var vout = new CudaDeviceVariable<Half>(cols);

        var pPos = (CudaDeviceVariable<int>)0;
        kernel.Run(qout.DevicePointer, kout.DevicePointer, vout.DevicePointer, xin.DevicePointer,
                   qwt.DevicePointer, qze.DevicePointer, qsc.DevicePointer,
                   kwt.DevicePointer, kze.DevicePointer, ksc.DevicePointer,
                   vwt.DevicePointer, vze.DevicePointer, vsc.DevicePointer,
                   inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, 0, pPos.DevicePointer);

        return ((Half[])qout, (Half[])kout, (Half[])vout);
    }

    private static Half[] MatVec(Half[] w, Half[] x, int n, int d)
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

    private static (Half[] q, Half[] k, Half[] v) Expected(Half[] xb, Half[] qw, Half[] kw, Half[] vw, int n, int d)
    {
        var q = MatVec(qw, xb, n, d);
        var k = MatVec(kw, xb, n, d);
        var v = MatVec(vw, xb, n, d);
        return (q, k, v);
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int rows = 128;
        const int cols = 128;
        var scales = Enumerable.Repeat(1f, 128).ToArray();
        var zeros = Enumerable.Repeat(0u, 128).ToArray();
        yield return new object[] { generator.NextArray(cols), //xb
                                    generator.NextIntArray(rows * cols, 0, 15).ToHalf(), //qw
                                    generator.NextIntArray(rows * cols, 0, 15).ToHalf(), //kw
                                    generator.NextIntArray(rows * cols, 0, 15).ToHalf(), //vw
                                    scales, zeros, rows, cols };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_QKVMatVecInt4(Half[] xb, Half[] qw, Half[] kw, Half[] vw,
                                   float[] scales, uint[] zeros, int rows, int cols)
    {
        var packed_qw = HostQWeight.Pack(qw, scales, zeros);
        var packed_kw = HostQWeight.Pack(kw, scales, zeros);
        var packed_vw = HostQWeight.Pack(vw, scales, zeros);

        var expected = Expected(xb, packed_qw.Unpack(), packed_kw.Unpack(), packed_vw.Unpack(), rows, cols);

        var actual = QKVMatVecInt4(xb, packed_qw, packed_kw, packed_vw, rows, cols);

        Assert.Equal(expected.q, actual.q);
        Assert.Equal(expected.k, actual.k);
        Assert.Equal(expected.v, actual.v);
    }
}
