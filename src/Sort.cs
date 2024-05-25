using ManagedCuda;

namespace libLlama2;

public class Sort : Module
{
    public Sort(CudaContext cudaContext, Config config) :
        base(cudaContext, "sort_kernel.ptx", "sort_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = config.vocabSize / 125;
    }

    public void Forward(CudaDeviceVariable<Half> logits, CudaDeviceVariable<int> indices, int size) =>
        kernel.Run(logits.DevicePointer, indices.DevicePointer, size);
}