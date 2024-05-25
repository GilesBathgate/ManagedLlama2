using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class VecMat : Module
{
    public VecMat(CudaContext cudaContext, Config config) :
        base(cudaContext, "vec_mat_kernel.ptx", "vec_mat_kernel")
    {
        var headSize = config.dim / config.numHeads;
        kernel.GridDimensions = new dim3(CeilDiv(headSize, 32), config.numHeads);
        kernel.BlockDimensions = new dim3(32, 32);
    }

    public void Forward(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> value,
                        CudaDeviceVariable<Half> attention, int headSize, int dim, int seqLength, SizeT layerOffset) =>
        kernel.Run(output.DevicePointer, attention.DevicePointer, value.OffsetPointer(layerOffset), headSize, seqLength, headSize, headSize, dim);
}