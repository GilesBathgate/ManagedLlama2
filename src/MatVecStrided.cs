using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class MatVecStrided : Module
{
    public MatVecStrided(CudaContext cudaContext, Config config) :
        base(cudaContext, "mat_vec_kernel.ptx", "mat_vec_strided_kernel")
    {
        kernel.GridDimensions = new dim3(CeilDiv(config.seq_len, 32), config.n_heads);
        kernel.BlockDimensions = new dim3(32, 32);
    }

    public void Forward(CudaDeviceVariable<Half> attention, CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key,
                        int headSize, int dim, int seqLength, SizeT layerOffset, float scale) =>
        kernel.Run(attention.DevicePointer, query.DevicePointer, key.OffsetPointer(layerOffset), headSize, seqLength, headSize, headSize, dim, seqLength, scale);

}