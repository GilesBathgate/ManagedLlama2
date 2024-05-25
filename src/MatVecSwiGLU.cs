using ManagedCuda;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class MatVecSwiGLU : Module
{
    public MatVecSwiGLU(CudaContext cudaContext, Config config) :
        base(cudaContext, "mat_vec_kernel.ptx", "mat_vec_swiglu_kernel")
    {
        kernel.GridDimensions = new dim3(CeilDiv(config.hidden_dim, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, QWeight g, QWeight u, int rows, int cols)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        kernel.Run(output.DevicePointer, input.DevicePointer,
                   g.Weight.DevicePointer, g.Zeros.DevicePointer, g.Scales.DevicePointer,
                   u.Weight.DevicePointer, u.Zeros.DevicePointer, u.Scales.DevicePointer,
                   rows, cols, zerosSize, scalesSize, weightsSize);
    }
}