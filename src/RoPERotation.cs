using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace libLlama2;

public class RoPERotation : Module
{
    private readonly int kvDim;

    public RoPERotation(CudaContext cudaContext, Config config) :
        base(cudaContext, "rope_kernel.ptx", "rope_kernel")
    {
        kvDim = config.dim * config.numKVHeads / config.numHeads;
        int headSize = config.dim / config.numHeads;
        kernel.GridDimensions = config.numHeads;
        kernel.BlockDimensions = headSize / 2;
    }

    public void Forward(CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key,
                        int numKVHeads, int headSize, int pos, SizeT layerOffset, float theta)
    {
        SizeT offset = layerOffset + pos * kvDim;
        kernel.Run(query.DevicePointer, key.OffsetPointer(offset), numKVHeads, headSize, pos, theta);
    }
}