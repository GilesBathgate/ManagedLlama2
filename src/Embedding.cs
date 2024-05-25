using ManagedCuda;

namespace libLlama2;

public class Embedding : Module
{
    public Embedding(CudaContext cudaContext, Config config) :
        base(cudaContext, "embedding_kernel.ptx", "embedding_kernel")
    {
        kernel.GridDimensions = CeilDiv(config.dim, 256);
        kernel.BlockDimensions = 256;
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> table, int size, CudaDeviceVariable<int> tokens, int pos) =>
        base.Forward(output.DevicePointer, table.DevicePointer, size, tokens.DevicePointer, pos);
}