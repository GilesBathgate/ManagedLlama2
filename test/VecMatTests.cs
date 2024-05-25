namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class VecMatTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public VecMatTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("vec_mat_kernel.ptx", "vec_mat_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] VecMat(Half[] att, Half[] vector, int dim, int n_heads, int seq_len)
    {
        var pos = seq_len - 1;
        var head_size = dim / n_heads;

        kernel.GridDimensions = new dim3(ceil_div(head_size, 32), n_heads);
        kernel.BlockDimensions = new dim3(32, 32);

        var m = (CudaDeviceVariable<Half>)att;
        var v = (CudaDeviceVariable<Half>)vector;

        var result = new CudaDeviceVariable<Half>(dim * head_size);

        var pPos = (CudaDeviceVariable<int>)pos;
        kernel.Run(result.DevicePointer, m.DevicePointer, v.DevicePointer, head_size, pPos.DevicePointer, head_size, head_size, dim, 1);

        return (Half[])result;
    }

    private Half[] Expected(Half[] att, Half[] v, int dim, int n_heads, int n_kv_heads, int seq_len)
    {
        const int loff = 0;

        var pos = seq_len - 1;
        var head_size = dim / n_heads;
        var kv_dim = dim * n_kv_heads / n_heads;
        var kv_mul = n_heads / n_kv_heads;

        // weighted sum of the values, store back into xb
        var xb = new float[dim * head_size];
        for (int h = 0; h < n_heads; h++)
        {
            int _xb = h * head_size;
            var _att = h * seq_len;
            for (int t = 0; t <= pos; t++)
            {
                // get the value vector for this head and at this timestep
                int _v = loff + t * kv_dim + h / kv_mul * head_size;
                // get the attention weight for this timestep
                float a = (float)att[_att + t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++)
                {
                    xb[_xb + i] += a * (float)v[_v + i];
                }
            }
        }
        return xb.ToHalf();
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int dim = 4096;
        const int n_heads = 32;
        const int seq_len = 16;
        yield return new object[] { generator.NextArray(dim), generator.NextArray(dim * n_heads), dim, n_heads, seq_len };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_VecMat(Half[] att, Half[] v, int dim, int n_heads, int seq_len)
    {
        var expected = Expected(att, v, dim, n_heads, n_heads, seq_len);

        var actual = VecMat(att, v, dim, n_heads, seq_len);

        Assert.Equal(expected, actual);
    }
}
