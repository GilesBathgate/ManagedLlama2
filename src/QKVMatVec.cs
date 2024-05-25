using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class QKVMatVec : Module
{
    private readonly int kvDim;

    public QKVMatVec(CudaContext cudaContext, Config config) :
        base(cudaContext, "mat_vec_kernel.ptx", "qkv_mat_vec_kernel")
    {
        kvDim = config.dim * config.numKVHeads / config.numHeads;
        kernel.GridDimensions = new dim3(CeilDiv(config.dim, 4), 3);
        kernel.BlockDimensions = new dim3(32, 4);
    }

    public void Forward(CudaDeviceVariable<Half> queryOutput, CudaDeviceVariable<Half> keyOutput, CudaDeviceVariable<Half> valueOutput,
                           CudaDeviceVariable<Half> input, QWeight query, QWeight key, QWeight value, int rows, int cols, SizeT layerOffset, int pos)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        SizeT offset = layerOffset + pos * kvDim;
        kernel.Run(queryOutput.DevicePointer, keyOutput.OffsetPointer(offset), valueOutput.OffsetPointer(offset), input.DevicePointer,
                   query.Weight.DevicePointer, query.Zeros.DevicePointer, query.Scales.DevicePointer,
                   key.Weight.DevicePointer, key.Zeros.DevicePointer, key.Scales.DevicePointer,
                   value.Weight.DevicePointer, value.Zeros.DevicePointer, value.Scales.DevicePointer,
                   rows, cols, zerosSize, scalesSize, weightsSize);
    }

}