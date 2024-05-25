using ManagedCuda;

namespace libLlama2;

public class CumulativeSum : Module
{
    public CumulativeSum(CudaContext cudaContext, Config config) :
        base(cudaContext, "cumulative_sum_kernel.ptx", "cumulative_sum_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = config.vocabSize / 32;
    }

    public void Forward(CudaDeviceVariable<Half> logits, int size) =>
        base.Forward(logits.DevicePointer, size);
}