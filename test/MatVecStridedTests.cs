namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.VectorTypes;

public class MatVecStridedTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public MatVecStridedTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_strided_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] MatVecStrided(Half[] q, Half[] k, int dim, int n_heads, int seq_len)
    {
        var head_size = dim / n_heads;
        var scale = 1.0f / MathF.Sqrt(head_size);

        var ip = (CudaDeviceVariable<Half>)q;
        var wt = (CudaDeviceVariable<Half>)k;
        var att = new CudaDeviceVariable<Half>(n_heads * seq_len);

        kernel.BlockDimensions = new dim3(32, 32);
        kernel.GridDimensions = new dim3(ceil_div(seq_len, 32), n_heads);

        kernel.Run(att.DevicePointer, ip.DevicePointer, wt.DevicePointer, head_size, seq_len, head_size, head_size, dim, seq_len, scale);
        return (Half[])att;
    }

    private Half[] Expected(Half[] q, Half[] k, int dim, int n_heads, int n_kv_heads, int seq_len)
    {
        const int loff = 0;

        var pos = seq_len - 1;
        var head_size = dim / n_heads;
        var scale = 1.0f / MathF.Sqrt(head_size);
        var kv_dim = dim * n_kv_heads / n_heads;
        var kv_mul = n_heads / n_kv_heads;

        var att = new float[n_heads * seq_len];

        for (int h = 0; h < n_heads; h++)
        {
            var _q = h * head_size;
            var _att = h * seq_len;
            for (int t = 0; t <= pos; t++)
            {
                var _k = loff + t * kv_dim + h / kv_mul * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += (float)q[_q + i] * (float)k[_k + i];
                }
                score *= scale;
                att[_att + t] = score;
            }
        }
        return att.ToHalf();
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
    public void Test_MatVecStrided(Half[] q, Half[] k, int dim, int n_heads, int seq_len)
    {
        var expected = Expected(q, k, dim, n_heads, n_heads, seq_len);

        var actual = MatVecStrided(q, k, dim, n_heads, seq_len);

        Assert.Equal(expected, actual);
    }
}
