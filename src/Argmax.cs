using ManagedCuda;

namespace libLlama2;

public class Argmax : Module
{
    public Argmax(CudaContext cudaContext) :
        base(cudaContext, "argmax_kernel.ptx", "argmax_kernel")
    {
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Forward(CudaDeviceVariable<Half> logits, int vocabSize, CudaDeviceVariable<int> tokens, int nextPos, bool generateToken) =>
        kernel.Run(logits.DevicePointer, vocabSize, tokens.DevicePointer, nextPos, generateToken);
}