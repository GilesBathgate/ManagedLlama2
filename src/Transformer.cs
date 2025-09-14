using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace libLlama2;

public class Transformer : ITransformer
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

    private readonly JsonStateMachine stateMachine;

    private readonly ConstraintGenerator constraintGenerator;

    public Transformer(string modelPath, string tokenizerPath = "tokenizer.bin", float temperature = 0.5f, float topP = 0.9f) :
        this(File.OpenRead(modelPath), tokenizerPath, temperature, topP)
    { }

    public Transformer(FileStream fileStream, string tokenizerPath, float temperature, float topP)
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);

        config = Config.LoadConfig(fileStream);

        kvDim = config.dim * config.numKVHeads / config.numHeads;

        if (kvDim != config.dim)
            throw new NotImplementedException("Differing kvDim dimention not currently supported.");

        weights = new TransformerWeights(config, fileStream);

        tokenizer = new Tokenizer(tokenizerPath, config.vocabSize);

        runstate = new RunState(cudaContext, ref config, kvDim);

        sampler = new Sampler(cudaContext, config, runstate, temperature, topP);

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

        stateMachine = new JsonStateMachine();

        constraintGenerator = new ConstraintGenerator(tokenizer, config.vocabSize);

        fileStream.Close();
    }

    public IEnumerable<Token> Generate(string prompt, int steps)
    {
        var promptTokens = tokenizer.Encode(prompt, true);

        runstate.tokens.CopyToDevice(promptTokens);

        var prev = promptTokens[0];
        for (int pos = 0; pos < steps; ++pos)
        {

            var nextPos = pos + 1;
            Forward(pos, nextPos);

            var generateToken = nextPos >= promptTokens.Length;

            var token = generateToken ? sampler.Sample(nextPos, generateToken) : promptTokens[nextPos];

            if (token < 3) break;

            var piece = tokenizer.Decode(prev, token);
            yield return new Token(token, piece);
            prev = token;
        }
    }

    public IEnumerable<Token> Chat(string system_prompt, IEnumerable<string> userInput)
    {
        var pos = 0;
        var prev = 0;
        foreach (var input in userInput)
        {
            var prompt = pos == 0 ? $"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST]"
                                  : $"[INST] {input} [/INST]";

            var promptTokens = tokenizer.Encode(prompt, true, false);

            runstate.tokens.CopyToDevice(promptTokens, pos);

            for (var userPos = pos + promptTokens.Length; pos < config.seqLength; ++pos)
            {
                var nextPos = pos + 1;
                Forward(pos, nextPos);

                var generateToken = nextPos >= userPos;

                runstate.constraints?.Dispose();
                runstate.constraints = null;
                if (generateToken)
                {
                    var (allow, constraint) = constraintGenerator.ConstrainedTokens(stateMachine);
                    if (constraint.Length > 0)
                    {
                        runstate.constraints = new CudaDeviceVariable<int>(constraint.Length);
                        runstate.constraints.CopyToDevice(constraint);
                        runstate.constraintIsAllow = allow;
                    }
                }

                var tokenId = sampler.Sample(nextPos, generateToken);

                if (generateToken)
                {

                    if (tokenId < 3) break;

                    var piece = tokenizer.Decode(prev, tokenId);
                    var token = new Token(tokenId, piece);
                    yield return token;

                    stateMachine.Process(token);
                    if (stateMachine.State == JsonStateMachine.JsonState.Complete)
                    {
                        stateMachine.Reset();
                        break;
                    }
                }
                prev = tokenId;
            }
            yield return new Token(0, Environment.NewLine);
            ++pos;
        }
    }

    private void MultiHeadAttention(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key, CudaDeviceVariable<Half> value,
                                    CudaDeviceVariable<Half> attention, int headSize, int dim, int seqLength, SizeT layerOffset, float scale)
    {
        matVecStrided.Forward(attention, query, key, headSize, dim, seqLength, layerOffset, scale);
        softmax.Forward(attention, seqLength);
        vecMat.Forward(output, value, attention, headSize, dim, seqLength, layerOffset);
    }

    private void Forward(int position, int nextPosition)
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

            MultiHeadAttention(runstate.xb, runstate.q, runstate.keyCache, runstate.valueCache, runstate.attention, headSize, config.dim, nextPosition, layerOffset, scale);

            matVecResidual.Forward(runstate.x, runstate.xb, layer.outputWeight, config.dim, config.dim);

            rmsNorm.Forward(runstate.xb, runstate.x, layer.rmsFeedForwardWeight, config.dim);

            matVecSwiGLU.Forward(runstate.h, runstate.xb, layer.gateWeight, layer.upWeight, config.dim, config.hiddenDim);

            matVecResidual.Forward(runstate.x, runstate.h, layer.downWeight, config.hiddenDim, config.dim);

        }

        rmsNorm.Forward(runstate.x, runstate.x, weights.rmsFinalWeight, config.dim);

        matVec.Forward(runstate.logits, runstate.x, weights.classifierWeights, config.dim, config.vocabSize);
    }
}
