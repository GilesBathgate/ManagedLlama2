using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class MatVec : Module
{
    public MatVec(CudaContext cudaContext, Config config) :
        base(cudaContext, "mat_vec_kernel.ptx", "mat_vec_kernel")
    {
        kernel.GridDimensions = new dim3(CeilDiv(config.vocab_size, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, CudaDeviceVariable<Half> matrix, int rows, int cols) =>
        kernel.Run(output.DevicePointer, input.DevicePointer, matrix.DevicePointer, rows, cols);
}