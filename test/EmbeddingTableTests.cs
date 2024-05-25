namespace libLlama2.UnitTests;

using ManagedCuda;

public class EmbeddingTableTests : IDisposable
{
    private CudaKernel kernel;

    private readonly CudaContext cudaContext;

    public EmbeddingTableTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("embedding_kernel.ptx", "copy_embedding_kernel");
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private int ceil_div(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    private Half[] Embed(Half[] table, int dim, int[] tokens)
    {
        kernel.GridDimensions = ceil_div(dim, 256);
        kernel.BlockDimensions = 256;

        var x = new CudaDeviceVariable<Half>(dim);
        var t = (CudaDeviceVariable<Half>)table;
        var tk = (CudaDeviceVariable<int>)tokens;
        var pPos = (CudaDeviceVariable<int>)0;
        kernel.Run(x.DevicePointer, t.DevicePointer, dim, tk.DevicePointer, pPos.DevicePointer);

        return (Half[])x;
    }

    [Fact]
    public void Test_Embed()
    {
        var generator = new Generator();

        var table = generator.NextArray(256);
        var tokens = new int[] { 0 };
        var x = Embed(table, table.Length, tokens);

        Assert.Equal(table, x);
    }
}
