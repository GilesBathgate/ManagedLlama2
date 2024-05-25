using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class MatVecResidual : Module
{
    public MatVecResidual(CudaContext cudaContext, Config config) :
        base(cudaContext, "mat_vec_kernel.ptx", "mat_vec_residual_int4_kernel")
    {
        kernel.GridDimensions = new dim3(CeilDiv(config.dim, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, QWeight matrix, int rows, int cols)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        kernel.Run(output.DevicePointer, input.DevicePointer,
                   matrix.Weight.DevicePointer, matrix.Zeros.DevicePointer, matrix.Scales.DevicePointer,
                   rows, cols, zerosSize, scalesSize, weightsSize);
    }

}