using ManagedCuda;

namespace libLlama2;

public class SoftmaxLogits : Module
{
    public SoftmaxLogits(CudaContext cudaContext) :
        base(cudaContext, "softmax_logits_kernel.ptx", "softmax_logits_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> logits, int size, float temperature, CudaDeviceVariable<int> indices) =>
        base.Forward(logits.DevicePointer, size, temperature, indices.DevicePointer);
}