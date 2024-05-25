using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace libLlama2;

public class Transformer
{
    private readonly Config config;

    private readonly CudaContext cudaContext;

    private readonly CudaKernel embeddingKernel;

    private readonly CudaKernel matVecKernel;

    private readonly CudaKernel matVecResidualKernel;

    private readonly CudaKernel matVecStridedKernel;

    private readonly CudaKernel matVecSwiGLUKernel;

    private readonly CudaKernel qkvMatVecKernel;

    private readonly CudaKernel rmsNormKernel;

    private readonly CudaKernel ropeKernel;

    private readonly CudaKernel softmaxKernel;

    private readonly CudaKernel vecMatKernel;

    private readonly int kvDim;

    private readonly RunState runstate;

    private readonly TransformerWeights weights;

    public Transformer(string modelPath) : this(File.OpenRead(modelPath)) {}

    public Transformer(FileStream fileStream)
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);

        config = LoadConfig(fileStream);

        kvDim = config.dim * config.n_kv_heads / config.n_heads;

        weights = CheckpointInitWeights(fileStream);

        runstate = InitRunState();

        embeddingKernel = InitEmbeddingKernel();

        rmsNormKernel = InitRmsNormKernel();

        qkvMatVecKernel = InitQKVMatVecKernel();

        ropeKernel = InitRopeKernel();

        matVecStridedKernel = InitMatVecStridedKernel();

        softmaxKernel = InitSoftmaxKernel();

        vecMatKernel = InitVecMatKernel();

        matVecResidualKernel = InitMatVecResidualKernel();

        matVecSwiGLUKernel = InitMatVecSwiGLUKernel();

        matVecKernel = InitMatVecKernel();

        fileStream.Close();
    }

    public void Run(int[] prompt)
    {
        var pos = 0;
        var tokens = new CudaDeviceVariable<int>(config.seq_len);
        tokens.CopyToDevice(prompt);

        RunNetwork(pos, tokens, pos + 1);
    }

    private static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;

    private TransformerWeights CheckpointInitWeights(FileStream fileStream)
    {
        var fileSize = fileStream.Length;
        using var memoryMappedFile = MemoryMappedFile.CreateFromFile(
            fileStream, null, fileSize, MemoryMappedFileAccess.Read, HandleInheritability.None, false);

        var remaining = fileSize - Config.Size;
        using var accessor = memoryMappedFile.CreateViewAccessor(
            Config.Size, remaining, MemoryMappedFileAccess.Read);

        long offset = 0;
        var weights = CheckpointInitWeights(accessor, ref offset);

        if (offset != remaining)
            throw new FileLoadException($"Failed to read file. offset: {offset} != length: {remaining}");

        return weights;
    }

    private TransformerWeights CheckpointInitWeights(MemoryMappedViewAccessor accessor, ref long offset) =>
        new()
        {
            tokenEmbeddingTable = ReadWeight(accessor, ref offset, config.vocab_size * config.dim),
            classifierWeights = ReadWeight(accessor, ref offset, config.vocab_size * config.dim),
            rmsFinalWeight = ReadWeight(accessor, ref offset, config.dim),
            layers = ReadLayers(accessor, ref offset)
        };

    private RunState InitRunState() =>
        new()
        {
            x = new CudaDeviceVariable<Half>(config.dim),
            xb = new CudaDeviceVariable<Half>(config.dim),
            hb = new CudaDeviceVariable<Half>(config.hidden_dim),
            hb2 = new CudaDeviceVariable<Half>(config.hidden_dim),
            q = new CudaDeviceVariable<Half>(config.dim),
            att = new CudaDeviceVariable<Half>(config.n_heads * config.dim),
            logits = new CudaDeviceVariable<Half>(config.vocab_size),
            keyCache = new CudaDeviceVariable<Half>(config.n_layers * config.seq_len * kvDim),
            valueCache = new CudaDeviceVariable<Half>(config.n_layers * config.seq_len * kvDim),
            logitsArray = new CudaDeviceVariable<float>(config.seq_len * config.vocab_size)
        };

    private CudaKernel InitEmbeddingKernel()
    {
        var kernel = cudaContext.LoadKernel("embedding_kernel.ptx", "embedding_kernel");
        kernel.GridDimensions = CeilDiv(config.dim, 256);
        kernel.BlockDimensions = 256;
        return kernel;
    }

    private CudaKernel InitMatVecKernel()
    {
        var kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_kernel");
        kernel.GridDimensions = new dim3(CeilDiv(config.vocab_size, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
        return kernel;
    }

    private CudaKernel InitMatVecResidualKernel()
    {
        var kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_residual_int4_kernel");
        kernel.GridDimensions = new dim3(CeilDiv(config.dim, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
        return kernel;
    }

    private CudaKernel InitMatVecStridedKernel()
    {
        var kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_strided_kernel");
        kernel.GridDimensions = new dim3(CeilDiv(config.seq_len, 32), config.n_heads);
        kernel.BlockDimensions = new dim3(32, 32);
        return kernel;
    }

    private CudaKernel InitMatVecSwiGLUKernel()
    {
        var kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "mat_vec_swiglu_kernel");
        kernel.GridDimensions = new dim3(CeilDiv(config.hidden_dim, 4), 1);
        kernel.BlockDimensions = new dim3(32, 4);
        return kernel;
    }

    private CudaKernel InitQKVMatVecKernel()
    {
        var kernel = cudaContext.LoadKernel("mat_vec_kernel.ptx", "qkv_mat_vec_kernel");
        kernel.GridDimensions = new dim3(CeilDiv(config.dim, 4), 3);
        kernel.BlockDimensions = new dim3(32, 4);
        return kernel;
    }

    private CudaKernel InitRmsNormKernel()
    {
        var kernel = cudaContext.LoadKernel("rmsnorm_kernel.ptx", "rmsnorm_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
        return kernel;
    }

    private CudaKernel InitRopeKernel()
    {
        var kernel = cudaContext.LoadKernel("rope_kernel.ptx", "rope_kernel");
        int headSize = config.dim / config.n_heads;
        kernel.GridDimensions = config.n_heads;
        kernel.BlockDimensions = headSize / 2;
        return kernel;
    }

    private CudaKernel InitSoftmaxKernel()
    {
        var kernel = cudaContext.LoadKernel("softmax_kernel.ptx", "softmax_kernel");
        kernel.GridDimensions = config.n_heads;
        kernel.BlockDimensions = 1024;
        return kernel;
    }

    private CudaKernel InitVecMatKernel()
    {
        var kernel = cudaContext.LoadKernel("vec_mat_kernel.ptx", "vec_mat_kernel");
        var headSize = config.dim / config.n_heads;
        kernel.GridDimensions = new dim3(CeilDiv(headSize, 32), config.n_heads);
        kernel.BlockDimensions = new dim3(32, 32);
        return kernel;
    }

    private static Config LoadConfig(FileStream fileStream)
    {
        using var reader = new BinaryReader(fileStream, Encoding.UTF8, true);
        var bytes = reader.ReadBytes(Config.Size);
        var config = MemoryMarshal.Cast<byte, Config>(bytes)[0];

        if (config.rope_theta != 10000.0f)
            throw new FileLoadException("Invalid model config");

        return config;
    }

    private static uint[] ReadIntArray(MemoryMappedViewAccessor accessor, ref long offset, int size)
    {
        var array = new uint[size];
        var read = accessor.ReadArray(offset, array, 0, size);
        offset += sizeof(uint) * read;
        return array;
    }

    private static QWeight ReadQWeight(MemoryMappedViewAccessor accessor, ref long offset, int rows, int cols)
    {
        const int group_size = 128;
        var scalesSize = CeilDiv(rows, group_size);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);

        var qWeight = new QWeight
        {
            Weight = ReadIntArray(accessor, ref offset, weightsSize * cols),
            Zeros = ReadIntArray(accessor, ref offset, zerosSize * cols),
            Scales = ReadWeight(accessor, ref offset, scalesSize * cols)
        };

        return qWeight;
    }

    private static Half[] ReadWeight(MemoryMappedViewAccessor accessor, ref long offset, int size)
    {
        const int sizeofHalf = 2; //float16 = 2 bytes;

        var array = new Half[size];
        var read = accessor.ReadArray(offset, array, 0, size);
        offset += sizeofHalf * read;
        return array;
    }

    private PerLayerWeight ReadLayer(MemoryMappedViewAccessor accessor, ref long offset) =>
        new()
        {
            queryWeight = ReadQWeight(accessor, ref offset, config.dim, config.dim),
            keyWeight = ReadQWeight(accessor, ref offset, config.dim, kvDim),
            valueWeight = ReadQWeight(accessor, ref offset, config.dim, kvDim),
            outputWeight = ReadQWeight(accessor, ref offset, config.dim, config.dim),

            upWeight = ReadQWeight(accessor, ref offset, config.dim, config.hidden_dim),
            gateWeight = ReadQWeight(accessor, ref offset, config.dim, config.hidden_dim),
            downWeight = ReadQWeight(accessor, ref offset, config.hidden_dim, config.dim),

            rmsAttentionWeight = ReadWeight(accessor, ref offset, config.dim),
            rmsFeedForwardWeight = ReadWeight(accessor, ref offset, config.dim)
        };

    private ICollection<PerLayerWeight> ReadLayers(MemoryMappedViewAccessor accessor, ref long offset)
    {
        var perLayerWeight = new List<PerLayerWeight>(config.n_layers);
        for (var i = 0; i < config.n_layers; ++i)
            perLayerWeight.Add(ReadLayer(accessor, ref offset));
        return perLayerWeight;
    }

    private void Embedding(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> table, int size, CudaDeviceVariable<int> tokens, int pos) =>
        embeddingKernel.Run(output.DevicePointer, table.DevicePointer, size, tokens.DevicePointer, pos);

    private void MatVec(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, CudaDeviceVariable<Half> matrix, int rows, int cols) =>
        matVecKernel.Run(output.DevicePointer, input.DevicePointer, matrix.DevicePointer, rows, cols);

    private void MatVecResidual(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, QWeight matrix, int rows, int cols)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        matVecResidualKernel.Run(output.DevicePointer, input.DevicePointer,
                                 matrix.Weight.DevicePointer, matrix.Zeros.DevicePointer, matrix.Scales.DevicePointer,
                                 rows, cols, zerosSize, scalesSize, weightsSize);
    }

    private void MatVecSwiGLU(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, QWeight g, QWeight u, int rows, int cols)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        matVecSwiGLUKernel.Run(output.DevicePointer, input.DevicePointer,
                               g.Weight.DevicePointer, g.Zeros.DevicePointer, g.Scales.DevicePointer,
                               u.Weight.DevicePointer, u.Zeros.DevicePointer, u.Scales.DevicePointer,
                               rows, cols, zerosSize, scalesSize, weightsSize);
    }

    private void MultiHeadAttention(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key, CudaDeviceVariable<Half> value,
                                    CudaDeviceVariable<Half> attention, int headSize, int dim, int seqLength, SizeT layerOffset, float scale)
    {
        matVecStridedKernel.Run(attention.DevicePointer, query.DevicePointer, key.DevicePointer + layerOffset, headSize, seqLength, headSize, headSize, dim, seqLength, scale);
        softmaxKernel.Run(attention.DevicePointer, seqLength);
        vecMatKernel.Run(output.DevicePointer, attention.DevicePointer, value.DevicePointer + layerOffset, headSize, seqLength, headSize, headSize, dim);
    }

    private void QKVMatVec(CudaDeviceVariable<Half> queryOutput, CudaDeviceVariable<Half> keyOutput, CudaDeviceVariable<Half> valueOutput,
                           CudaDeviceVariable<Half> input, QWeight query, QWeight key, QWeight value, int rows, int cols, SizeT layerOffset, int pos)
    {
        var scalesSize = CeilDiv(rows, 128);
        var weightsSize = CeilDiv(rows, 32) * 4;
        var zerosSize = CeilDiv(scalesSize, 8);
        var offset = layerOffset + pos * cols;
        qkvMatVecKernel.Run(queryOutput.DevicePointer, keyOutput.DevicePointer + offset, valueOutput.DevicePointer + offset, input.DevicePointer,
                            query.Weight.DevicePointer, query.Zeros.DevicePointer, query.Scales.DevicePointer,
                            key.Weight.DevicePointer, key.Zeros.DevicePointer, key.Scales.DevicePointer,
                            value.Weight.DevicePointer, value.Zeros.DevicePointer, value.Scales.DevicePointer,
                            rows, cols, zerosSize, scalesSize, weightsSize);
    }

    private void RMSNorm(CudaDeviceVariable<Half> output, CudaDeviceVariable<Half> input, CudaDeviceVariable<Half> weights, int size, float eps = 1e-5f) =>
        rmsNormKernel.Run(output.DevicePointer, input.DevicePointer, weights.DevicePointer, size, eps);

    private void RoPERotation(CudaDeviceVariable<Half> query, CudaDeviceVariable<Half> key,
                              int numHeads, int numKVHeads, int headSize, int pos, SizeT layerOffset, float theta)
    {
        var offset = layerOffset + pos * numKVHeads * headSize;
        ropeKernel.Run(query.DevicePointer, key.DevicePointer + offset, headSize, numHeads, numKVHeads, pos, theta);
    }

    private void RunNetwork(int pos, CudaDeviceVariable<int> tokens, int seq_len_bin)
    {
        var p = config;
        var dim = config.dim;
        var s = runstate;
        var head_size =  dim / p.n_heads;
        var scale = 1.0f / MathF.Sqrt(head_size);

        Embedding(s.x, weights.tokenEmbeddingTable, dim, tokens, pos);

        foreach (var (i, l) in weights.layers.Enumerate())
        {
            RMSNorm(s.xb, s.x, l.rmsAttentionWeight, dim);

            SizeT loff = i * config.seq_len * kvDim;

            if (dim == kvDim) {
                QKVMatVec(s.q, s.keyCache, s.valueCache, s.xb, l.queryWeight, l.keyWeight, l.valueWeight, dim, dim, loff, pos);
            } else {
                throw new NotImplementedException();
            }

            RoPERotation(s.q, s.keyCache, p.n_heads, p.n_kv_heads, head_size, pos, loff, p.rope_theta);

            MultiHeadAttention(s.xb, s.q, s.keyCache, s.valueCache, s.att, head_size, dim, seq_len_bin, loff, scale);

            MatVecResidual(s.x, s.xb, l.outputWeight, dim, dim);

            RMSNorm(s.xb, s.x, l.rmsFeedForwardWeight, dim);

            MatVecSwiGLU(s.hb, s.xb, l.gateWeight, l.upWeight, dim, p.hidden_dim);

            MatVecResidual(s.x, s.hb, l.downWeight, p.hidden_dim, dim);

        }

        RMSNorm(s.x, s.x, weights.rmsFinalWeight, dim);

        MatVec(s.logits, s.x, weights.classifierWeights, p.dim, p.vocab_size);

    }

}
