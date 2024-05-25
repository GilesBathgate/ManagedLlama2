namespace libLlama2.UnitTests;

using ManagedCuda;
using ManagedCuda.BasicTypes;

public class RoPETests : IDisposable
{
    private readonly CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public RoPETests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("rope_kernel.ptx", "rope_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private void RoPE(ref Half[] q, ref Half[] k, int dim, int seq_len, int n_heads, int n_kv_heads, int pos, float theta = 10000.0f)
    {
        var sq = (CudaDeviceVariable<Half>)q;
        var sk = (CudaDeviceVariable<Half>)k;

        const int l = 0;
        int kv_dim = (dim * n_kv_heads) / n_heads;
        int head_size = dim / n_heads;
        int loff = l * seq_len * kv_dim;
        SizeT offset = l + loff + pos * kv_dim;

        kernel.GridDimensions = n_heads;
        kernel.BlockDimensions = head_size / 2;

        kernel.Run(sq.DevicePointer, sk.DevicePointer + (offset * sk.TypeSize), n_kv_heads, head_size, pos, theta);

        q = (Half[])sq;
        k = (Half[])sk;
    }

    private static void Expected(ref Half[] q, ref Half[] k, int dim, int seq_len, int n_heads, int n_kv_heads, int pos, float theta = 10000.0f)
    {
        const int l = 0;
        int kv_dim = (dim * n_kv_heads) / n_heads;
        int head_size = dim / n_heads;
        int loff = l * seq_len * kv_dim;
        int _k = l + loff + pos * kv_dim;
        int size = head_size / 2;

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int h = 0; h < n_heads; ++h)
        {
            for (int t = 0; t < size; ++t)
            {
                int i = h * head_size + t;
                int j = size + i;
                int head_dim = i * 2 % head_size;
                float freq = 1.0f / MathF.Pow(theta, head_dim / (float)head_size);
                float val = pos * freq;
                float fci = MathF.Sin(val);
                float fcr = MathF.Cos(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++)
                {
                    if (v == 0)
                    {
                        var v0 = (float)q[i];
                        var v1 = (float)q[j];
                        q[i] = (Half)(v0 * fcr - v1 * fci);
                        q[j] = (Half)(v0 * fci + v1 * fcr);
                    }
                    else
                    {
                        var v0 = (float)k[_k + i];
                        var v1 = (float)k[_k + j];
                        k[_k + i] = (Half)(v0 * fcr - v1 * fci);
                        k[_k + j] = (Half)(v0 * fci + v1 * fcr);
                    }
                }
            }
        }
    }

    private static (Half[] q, Half[] k) Copy(Half[] q, Half[] k)
    {
        var copy_q = new Half[q.Length];
        var copy_k = new Half[k.Length];
        q.CopyTo(copy_q, 0);
        k.CopyTo(copy_k, 0);
        return (copy_q, copy_k);
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        const int seq_len = 20;
        const int dim = 128;
        const int heads = 32;
        var q = generator.NextArray(dim);
        var k = generator.NextArray(dim * seq_len);
        for (int pos = 0; pos < seq_len - 1; ++pos)
            yield return new object[] { q, k, dim, seq_len, heads, pos };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_RoPE(Half[] q, Half[] k, int dim, int seq_len, int heads, int pos)
    {
        var (actual_q, actual_k) = Copy(q, k);
        RoPE(ref actual_q, ref actual_k, dim, seq_len, heads, heads, pos);

        var (expected_q, expected_k) = Copy(q, k);
        Expected(ref expected_q, ref expected_k, dim, seq_len, heads, heads, pos);

        Assert.Equal(expected_q, actual_q);
        Assert.Equal(expected_k, actual_k);
    }
}
