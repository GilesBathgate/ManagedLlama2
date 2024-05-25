using ManagedCuda;

namespace libLlama2;

public class RunState
{
    public required CudaDeviceVariable<Half> x; // activation at current time stamp (dim,)

    public required CudaDeviceVariable<Half> xb; // same, but inside a residual branch (dim,)

    public required CudaDeviceVariable<Half> hb; // buffer for hidden dimension in the ffn (hidden_dim,)

    public required CudaDeviceVariable<Half> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)

    public required CudaDeviceVariable<Half> q; // query (dim,)

    public required CudaDeviceVariable<Half> att; // buffer for scores/attention values (n_heads, seq_len)

    public required CudaDeviceVariable<Half> logits; // output logits

    public required CudaDeviceVariable<Half> keyCache;   // (layer, seq_len, kv_dim)

    public required CudaDeviceVariable<Half> valueCache; // (layer, seq_len, kv_dim)

    public required CudaDeviceVariable<float> logitsArray;  // array of output logits used to compute perplexity (seq_len, vocab_size)
}