using ManagedCuda;

namespace libLlama2;

public class Sampler : ISampler
{
    private readonly RunState runstate;

    private readonly Argmax argmax;

    private readonly SoftmaxLogits softmax;

    private readonly CumulativeSum cumulativeSum;

    private readonly SampleTopP sampleTopP;

    private readonly Sort sort;

    private readonly float temperature;

    private readonly float topP;

    private readonly Config config;

    private readonly Random random = new();

    public Sampler(CudaContext context, Config config, RunState runstate, float temperature, float topP)
    {
        this.config = config;
        this.runstate = runstate;
        this.temperature = temperature;
        this.topP = topP;
        argmax = new Argmax(context);
        softmax = new SoftmaxLogits(context);
        cumulativeSum = new CumulativeSum(context, config);
        sampleTopP = new SampleTopP(context);
        sort = new Sort(context, config);
    }

    public int Sample(int nextPosition, bool generateToken)
    {
        var coin = random.NextSingle();

        if (temperature == 0.0f || !generateToken)
        {
            argmax.Forward(runstate.logits, config.vocabSize, runstate.tokens, nextPosition, generateToken);
        }
        else
        {
            softmax.Forward(runstate.logits, config.vocabSize, temperature, runstate.indices);

            float threshold;
            if (topP <= 0 || topP >= 1)
            {
                threshold = coin;
            }
            else
            {
                sort.Forward(runstate.logits, runstate.indices, config.vocabSize);

                threshold = coin * topP;
            }

            cumulativeSum.Forward(runstate.logits, config.vocabSize);

            sampleTopP.Forward(runstate.logits, runstate.indices, config.vocabSize, threshold, runstate.tokens, nextPosition);
        }

        return runstate.tokens[nextPosition];
    }

}