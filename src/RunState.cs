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
        h = new CudaDeviceVariable<Half>(config.hidden_dim);
        q = new CudaDeviceVariable<Half>(config.dim);
        attention = new CudaDeviceVariable<Half>(config.n_heads * config.dim);
        logits = new CudaDeviceVariable<Half>(config.vocab_size);
        keyCache = new CudaDeviceVariable<Half>(config.n_layers * config.seq_len * kvDim);
        valueCache = new CudaDeviceVariable<Half>(config.n_layers * config.seq_len * kvDim);
        logitsArray = new CudaDeviceVariable<float>(config.seq_len * config.vocab_size);
        tokens = new CudaDeviceVariable<int>(config.seq_len);
    }
}