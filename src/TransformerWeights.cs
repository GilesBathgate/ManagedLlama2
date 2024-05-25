using System.IO.MemoryMappedFiles;
using ManagedCuda;

namespace libLlama2;

public class TransformerWeights
{
    public readonly CudaDeviceVariable<Half> tokenEmbeddingTable; // token embedding table

    public readonly CudaDeviceVariable<Half> classifierWeights; // classifier weights for the logits, on the last layer

    public readonly CudaDeviceVariable<Half> rmsFinalWeight; // final rmsnorm

    public readonly ICollection<PerLayerWeight> layers;

    private readonly Config config;

    private readonly int kvDim;

    public TransformerWeights(Config config, FileStream fileStream)
    {
        this.config = config;
        kvDim = config.dim * config.n_kv_heads / config.n_heads;

        var fileSize = fileStream.Length;
        using var memoryMappedFile = MemoryMappedFile.CreateFromFile(
            fileStream, null, fileSize, MemoryMappedFileAccess.Read, HandleInheritability.None, false);

        var remaining = fileSize - Config.Size;
        using var accessor = memoryMappedFile.CreateViewAccessor(
            Config.Size, remaining, MemoryMappedFileAccess.Read);

        long offset = 0;
        tokenEmbeddingTable = ReadWeight(accessor, ref offset, config.vocab_size * config.dim);
        classifierWeights = ReadWeight(accessor, ref offset, config.vocab_size * config.dim);
        rmsFinalWeight = ReadWeight(accessor, ref offset, config.dim);
        layers = ReadLayers(accessor, ref offset);

        if (offset != remaining)
            throw new FileLoadException($"Failed to read file. offset: {offset} != length: {remaining}");
    }

    private static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;

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
}