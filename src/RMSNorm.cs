using ManagedCuda;

namespace libLlama2;

public class RMSNorm : Module
{
    public RMSNorm(CudaContext cudaContext) :
        base(cudaContext, "rmsnorm_kernel.ptx", "rmsnorm_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, CudaDeviceVariable<Half> weights, int size, float eps = 1e-5f) =>
        kernel.Run(output.DevicePointer, input.DevicePointer, weights.DevicePointer, size, eps);
}