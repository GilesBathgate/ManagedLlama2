using ManagedCuda;

namespace libLlama2;

public class Softmax : Module
{
    public Softmax(CudaContext cudaContext, Config config) :
        base(cudaContext, "softmax_kernel.ptx", "softmax_kernel")
    {
        kernel.GridDimensions = config.n_heads;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> attention, int seqLength) =>
        kernel.Run(attention.DevicePointer, seqLength);
}