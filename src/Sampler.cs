using ManagedCuda;

namespace libLlama2;

public class Sampler : ISampler
{
    private readonly RunState runstate;

    private readonly CudaKernel argmaxKernel;

    private const float temperature = 0.0f;

    private readonly Config config;

    public Sampler(CudaContext context, Config config, RunState runstate) {
        this.config = config;
        this.runstate = runstate;
        argmaxKernel = context.LoadKernel("argmax_kernel.ptx", "argmax_kernel");
        argmaxKernel.GridDimensions = 1;
        argmaxKernel.BlockDimensions = 1024;
    }

    public int Sample(int pos, bool generateToken)
    {
        var nextPos = pos + 1;
        if (temperature == 0.0f || !generateToken)
        {
            argmaxKernel.Run(runstate.logits.DevicePointer, config.vocab_size, runstate.tokens.DevicePointer, nextPos, generateToken);
        }

        return runstate.tokens[nextPos];
    }

}