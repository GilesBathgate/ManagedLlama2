using ManagedCuda;

namespace libLlama2;

public class RunState
{
    public readonly CudaDeviceVariable<Half> x; // activation at current time stamp (dim,)

    public readonly CudaDeviceVariable<Half> xb; // same, but inside a residual branch (dim,)

    public readonly CudaDeviceVariable<Half> h; // buffer for hidden dimension in the ffn (hidden_dim,)

    public readonly CudaDeviceVariable<Half> q; // query (dim,)

    public readonly CudaDeviceVariable<Half> attention; // buffer for scores/attention values (n_heads, seq_len)

    public readonly CudaDeviceVariable<Half> logits; // output logits

    public readonly CudaDeviceVariable<Half> keyCache;   // (layer, seq_len, kv_dim)

    public readonly CudaDeviceVariable<Half> valueCache; // (layer, seq_len, kv_dim)

    public readonly CudaDeviceVariable<float> logitsArray;  // array of output logits used to compute perplexity (seq_len, vocab_size)

    public readonly CudaDeviceVariable<int> tokens;

    public RunState(Config config, int kvDim)
    {
        x = new CudaDeviceVariable<Half>(config.dim);
        xb = new CudaDeviceVariable<Half>(config.dim);
        h = new CudaDeviceVariable<Half>(config.hiddenDim);
        q = new CudaDeviceVariable<Half>(config.dim);
        attention = new CudaDeviceVariable<Half>(config.numHeads * config.dim);
        logits = new CudaDeviceVariable<Half>(config.vocabSize);
        keyCache = new CudaDeviceVariable<Half>(config.numLayers * config.seqLength * kvDim);
        valueCache = new CudaDeviceVariable<Half>(config.numLayers * config.seqLength * kvDim);
        logitsArray = new CudaDeviceVariable<float>(config.seqLength * config.vocabSize);
        tokens = new CudaDeviceVariable<int>(config.seqLength);
    }
}