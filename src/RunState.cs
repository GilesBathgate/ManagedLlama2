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

    public readonly CudaDeviceVariable<int> tokens;

    public readonly CudaDeviceVariable<int> indices;

    public CudaDeviceVariable<int>? constraints;

    public RunState(CudaContext cudaContext, ref Config config, int kvDim)
    {
        int CalculateMaxUsableSequence(Config config)
        {
            const int seqLength = 1;
            const int sizeofHalf = 2;
            const int kernelMem = 8 * 1024 * 1024; // 8 MiB reserve for kernels etc.
            var remaining = cudaContext.GetFreeDeviceMemorySize() - kernelMem;
            var required = config.numLayers * seqLength * kvDim * sizeofHalf * 2 +
                           seqLength * sizeof(int) +
                           config.vocabSize * sizeof(int);
            return Math.Min(remaining / required, config.seqLength);
        }

        x = new CudaDeviceVariable<Half>(config.dim);
        xb = new CudaDeviceVariable<Half>(config.dim);
        h = new CudaDeviceVariable<Half>(config.hiddenDim);
        q = new CudaDeviceVariable<Half>(config.dim);
        attention = new CudaDeviceVariable<Half>(config.numHeads * config.dim);
        logits = new CudaDeviceVariable<Half>(config.vocabSize);

        config.seqLength = CalculateMaxUsableSequence(config);

        keyCache = new CudaDeviceVariable<Half>(config.numLayers * config.seqLength * kvDim);
        valueCache = new CudaDeviceVariable<Half>(config.numLayers * config.seqLength * kvDim);
        tokens = new CudaDeviceVariable<int>(config.seqLength);
        indices = new CudaDeviceVariable<int>(config.vocabSize);
    }
}