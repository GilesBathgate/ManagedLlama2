using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace libLlama2;

public class Transformer
{
    private readonly Config config;

    private readonly CudaContext cudaContext;

    private readonly Embedding embedding;

    private readonly MatVec matVec;

    private readonly MatVecResidual matVecResidual;

    private readonly MatVecStrided matVecStrided;

    private readonly MatVecSwiGLU matVecSwiGLU;

    private readonly QKVMatVec qkvMatVec;

    private readonly RMSNorm rmsNorm;

    private readonly RoPERotation rope;

    private readonly Softmax softmax;

    private readonly VecMat vecMat;

    private readonly int kvDim;

    private readonly RunState runstate;

    private readonly TransformerWeights weights;

    private readonly ITokenizer tokenizer;

    private readonly ISampler sampler;

    public Transformer(string modelPath) : this(File.OpenRead(modelPath)) {}

    public Transformer(FileStream fileStream)
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);

        config = Config.LoadConfig(fileStream);

        kvDim = config.dim * config.numKVHeads / config.numHeads;

        if (kvDim != config.dim)
            throw new NotImplementedException("Differing kvDim dimention not currently supported.");

        weights = new TransformerWeights(config, fileStream);

        tokenizer = new Tokenizer("tokenizer.bin", config.vocabSize);

        runstate = new RunState(config, kvDim);

        sampler = new Sampler(cudaContext, config, runstate);

        embedding = new Embedding(cudaContext, config);

        rmsNorm = new RMSNorm(cudaContext);

        qkvMatVec = new QKVMatVec(cudaContext, config);

        rope = new RoPERotation(cudaContext, config);

        matVecStrided = new MatVecStrided(cudaContext, config);

        softmax = new Softmax(cudaContext, config);

        vecMat = new VecMat(cudaContext, config);

        matVecResidual = new MatVecResidual(cudaContext, config);

        matVecSwiGLU = new MatVecSwiGLU(cudaContext, config);

        matVec = new MatVec(cudaContext, config);

        fileStream.Close();
    }

    public IList<Half[]> testLogits = new List<Half[]>();

    public IEnumerable<string> Generate(string prompt, int steps)
    {
        var promptTokens = tokenizer.Encode(prompt, true);

        runstate.tokens.CopyToDevice(promptTokens);

        var prev = promptTokens[0];
        for (int pos = 0; pos < steps; ++pos)
        {

            var seq_len_bin = pos + 1;
            Forward(pos, seq_len_bin);

            testLogits.Add(runstate.logits);

            var generateToken = pos >= promptTokens.Length - 1;

            var token = generateToken ? sampler.Sample(pos, generateToken) : promptTokens[pos + 1];

            if (token == 1) break;

            var piece = tokenizer.Decode(prev, token);
            yield return piece;
            prev = token;
        }
    }

    private void MultiHeadAttention(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key, CudaDeviceVariable<Half> value,
                                    CudaDeviceVariable<Half> attention, int headSize, int dim, int seqLength, SizeT layerOffset, float scale)
    {
        matVecStrided.Forward(attention, query, key, headSize, dim, seqLength, layerOffset, scale);
        softmax.Forward(attention, seqLength);
        vecMat.Forward(output, value, attention, headSize, dim, seqLength, layerOffset);
    }

    private void Forward(int position, int seq_len_bin)
    {
        var headSize = config.dim / config.numHeads;
        var scale = 1.0f / MathF.Sqrt(headSize);

        embedding.Forward(runstate.x, weights.tokenEmbeddingTable, config.dim, runstate.tokens, position);

        foreach (var (i, layer) in weights.layers.Enumerate())
        {
            rmsNorm.Forward(runstate.xb, runstate.x, layer.rmsAttentionWeight, config.dim);

            SizeT layerOffset = i * config.seqLength * kvDim;

            qkvMatVec.Forward(runstate.q, runstate.keyCache, runstate.valueCache, runstate.xb, layer.queryWeight, layer.keyWeight, layer.valueWeight, config.dim, config.dim, layerOffset, position);

            rope.Forward(runstate.q, runstate.keyCache, config.numKVHeads, headSize, position, layerOffset, config.ropeTheta);

            MultiHeadAttention(runstate.xb, runstate.q, runstate.keyCache, runstate.valueCache, runstate.attention, headSize, config.dim, seq_len_bin, layerOffset, scale);

            matVecResidual.Forward(runstate.x, runstate.xb, layer.outputWeight, config.dim, config.dim);

            rmsNorm.Forward(runstate.xb, runstate.x, layer.rmsFeedForwardWeight, config.dim);

            matVecSwiGLU.Forward(runstate.h, runstate.xb, layer.gateWeight, layer.upWeight, config.dim, config.hiddenDim);

            matVecResidual.Forward(runstate.x, runstate.h, layer.downWeight, config.hiddenDim, config.dim);

        }

        rmsNorm.Forward(runstate.x, runstate.x, weights.rmsFinalWeight, config.dim);

        matVec.Forward(runstate.logits, runstate.x, weights.classifierWeights, config.dim, config.vocabSize);
    }
}
