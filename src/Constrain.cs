using ManagedCuda;

namespace libLlama2;

public class Constrain : Module
{
    public Constrain(CudaContext cudaContext) :
        base(cudaContext, "sample_constrained_kernel.ptx", "sample_constrained_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> logits, int size, CudaDeviceVariable<int> constraint, bool allow) =>
        base.Forward(logits.DevicePointer, size, constraint.DevicePointer, constraint.Size, allow);
}
