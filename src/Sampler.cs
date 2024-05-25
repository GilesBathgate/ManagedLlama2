using ManagedCuda;

namespace libLlama2;

public class Sampler : ISampler
{
    private readonly RunState runstate;

    private readonly Argmax argmax;

    private const float temperature = 0.0f;

    private readonly Config config;

    public Sampler(CudaContext context, Config config, RunState runstate) {
        this.config = config;
        this.runstate = runstate;
        argmax = new Argmax(context);
    }

    public int Sample(int nextPosition, bool generateToken)
    {
        if (temperature == 0.0f || !generateToken)
        {
            argmax.Forward(runstate.logits, config.vocabSize, runstate.tokens, nextPosition, generateToken);
        }

        return runstate.tokens[nextPosition];
    }

}