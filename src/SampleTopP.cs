using ManagedCuda;

namespace libLlama2;

public class SampleTopP : Module
{
    public SampleTopP(CudaContext cudaContext) :
        base(cudaContext, "sample_top_p_kernel.ptx", "sample_top_p_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> logits, CudaDeviceVariable<int> indices, int size, float threshold, CudaDeviceVariable<int> tokens, int nextPos) =>
        kernel.Run(logits.DevicePointer, indices.DevicePointer, size, threshold, tokens.DevicePointer, nextPos);
}