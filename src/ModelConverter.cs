using System.Collections;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Razorvine.Pickle;

namespace libLlama2;

public class ModelConverter : IDisposable
{
    private readonly List<string> modelPaths;

    private readonly string configPath;

    private static readonly int[] orderMap = { 0, 2, 4, 6, 1, 3, 5, 7 };

    public enum ModelType { Llama2_AWQ_7b, Llama2_AWQ_13b }

    public ModelConverter(bool download) : this(ModelType.Llama2_AWQ_7b, download)
    { }

    public ModelConverter(ModelType modelType, bool download) : this("pytorch_model.pt", "config.json", modelType, download)
    { }

    public ModelConverter(string modelPath, string configPath, ModelType modelType, bool download = false)
    {
        if (download)
        {
            Downloader downloader;
            if (modelType == ModelType.Llama2_AWQ_13b)
            {
                modelPaths = GetTempPaths(modelPath);
                downloader = new Model13bDownloader(modelPaths, configPath);
            }
            else
            {
                modelPaths = new List<string> { modelPath };
                downloader = new Model7bDownloader(modelPath, configPath);
            }

            downloader.Download();
        }
        else
        {
            if (!File.Exists(modelPath))
                throw new ArgumentException("Download set to false and model path does not exist");

            if (!File.Exists(configPath))
                throw new ArgumentException("Download set to false and config path does not exist");

            modelPaths = new List<string> { modelPath };
        }

        this.configPath = configPath;
    }

    public void Dispose()
    {
        foreach (var file in modelPaths)
            File.Delete(file);
        File.Delete(configPath);
    }

    private static List<string> GetTempPaths(string modelPath)
    {
        var paths = new List<string>();
        for (var i = 0; i < 3; ++i)
            paths.Add(Path.ChangeExtension(modelPath, $".00{i + 1}.pt"));
        return paths;
    }

    private class JsonConfig
    {
        [JsonPropertyName("hidden_size")]
        public int HiddenSize { get; set; }

        [JsonPropertyName("intermediate_size")]
        public int IntermediateSize { get; set; }

        [JsonPropertyName("max_position_embeddings")]
        public int MaxPositionEmbeddings { get; set; }

        [JsonPropertyName("num_attention_heads")]
        public int NumAttentionHeads { get; set; }

        [JsonPropertyName("num_hidden_layers")]
        public int NumHiddenLayers { get; set; }

        [JsonPropertyName("num_key_value_heads")]
        public int NumKeyValueHeads { get; set; }

        [JsonPropertyName("rope_theta")]
        public float? RopeTheta { get; set; }

        [JsonPropertyName("vocab_size")]
        public int VocabSize { get; set; }

        public static Config Convert(string configPath)
        {
            using var configStream = File.OpenRead(configPath);
            var jsonConfig = JsonSerializer.Deserialize<JsonConfig>(configStream)
                ?? throw new FileLoadException("Config could not be loaded");

            const float defaultRopeTheta = 10000.0f;

            return new Config
            {
                dim = jsonConfig.HiddenSize,
                hiddenDim = jsonConfig.IntermediateSize,
                numLayers = jsonConfig.NumHiddenLayers,
                numHeads = jsonConfig.NumAttentionHeads,
                numKVHeads = jsonConfig.NumKeyValueHeads,
                vocabSize = jsonConfig.VocabSize,
                seqLength = jsonConfig.MaxPositionEmbeddings,
                ropeTheta = jsonConfig.RopeTheta ?? defaultRopeTheta
            };
        }
    }

    public enum Format { v0, v1, v2 }

    public void Convert(string outputPath, Format format = Format.v0) =>
        Convert(File.Create(outputPath), format);

    public void Convert(Stream outputStream, Format format = Format.v0)
    {
        Console.WriteLine($"Converting");
        using var archives = new DisposableList<ZipArchive>();

        var config = JsonConfig.Convert(configPath);
        Config.SaveConfig(outputStream, config);

        var dict = new Dictionary<string, Tensor>();
        foreach (var path in modelPaths)
        {
            var archive = new ZipArchive(File.OpenRead(path), ZipArchiveMode.Read);
            UpdateStateDictionary(archive, dict);
            archives.Add(archive);
        }

        var embed = dict["model.embed_tokens.weight"];
        embed.CopyTo(outputStream);

        var head = dict["lm_head.weight"];
        head.CopyTo(outputStream);

        var norm = dict["model.norm.weight"];
        norm.CopyTo(outputStream);

        int kvDim = config.dim * config.numKVHeads / config.numHeads;

        for (var i = 0; i < config.numLayers; ++i)
        {
            Console.Write(".");
            var layerName = $"model.layers.{i}";
            RepackQWeightByName(outputStream, dict, layerName, "self_attn.q_proj", config.dim, config.dim, format);
            RepackQWeightByName(outputStream, dict, layerName, "self_attn.k_proj", config.dim, kvDim, format);
            RepackQWeightByName(outputStream, dict, layerName, "self_attn.v_proj", config.dim, kvDim, format);
            RepackQWeightByName(outputStream, dict, layerName, "self_attn.o_proj", config.dim, config.dim, format);

            RepackQWeightByName(outputStream, dict, layerName, "mlp.up_proj", config.dim, config.hiddenDim, format);
            RepackQWeightByName(outputStream, dict, layerName, "mlp.gate_proj", config.dim, config.hiddenDim, format);
            RepackQWeightByName(outputStream, dict, layerName, "mlp.down_proj", config.hiddenDim, config.dim, format);

            var layerNorm = dict[$"{layerName}.input_layernorm.weight"];
            layerNorm.CopyTo(outputStream);

            var ffnLayerNorm = dict[$"{layerName}.post_attention_layernorm.weight"];
            ffnLayerNorm.CopyTo(outputStream);
        }
        Console.WriteLine("done");
    }

    private static void UpdateStateDictionary(ZipArchive archive, IDictionary<string, Tensor> state)
    {
        var pickle = archive.Entries.First(e => e.Name.EndsWith("data.pkl"));
        using var unpickler = new TorchUnpickler(archive);
        var table = (Hashtable)unpickler.load(pickle.Open());
        foreach (DictionaryEntry entry in table)
        {
            if (entry.Value is Tensor tensor)
            {
                state.Add((string)entry.Key, tensor);
            }
        }
    }

    protected static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;

    private static void RepackQWeightByName(Stream outputStream, IDictionary<string, Tensor> dict, string layerName, string weightName, int height, int width, Format format)
    {
        const int sizeofHalf = 2;
        const int groupSize = 128;

        var scalesSize = CeilDiv(height, groupSize);
        var weightsSize = CeilDiv(height, 8);
        var zerosSize = CeilDiv(scalesSize, 8);

        int originalQWeightBytes;
        int originalQZerosBytes;
        int originalScalesBytes;
        switch (format)
        {
            case Format.v0:
            {
                var originalQWeightWidth = CeilDiv(width, 8);
                var originalQZerosSize = CeilDiv(height, groupSize);
                originalQWeightBytes = originalQWeightWidth * height * sizeof(int);
                originalQZerosBytes = originalQWeightWidth * originalQZerosSize * sizeof(int);
                originalScalesBytes = originalQZerosSize * width * sizeofHalf;
            }
            break;
            case Format.v1:
            {
                var originalScalesHeight = zerosSize * 8;
                originalQWeightBytes = weightsSize * width * sizeof(int);
                originalQZerosBytes = zerosSize * width * sizeof(int);
                originalScalesBytes = originalScalesHeight * width * sizeofHalf;
            }
            break;
            default:
                throw new FormatException("Format is not supported");
        }

        var qWeightTensor = dict[$"{layerName}.{weightName}.qweight"];
        var qweights = qWeightTensor.ReadBytes(originalQWeightBytes);

        var qZerosTensor = dict[$"{layerName}.{weightName}.qzeros"];
        var qzeros = qZerosTensor.ReadBytes(originalQZerosBytes);

        var scalesTensor = dict[$"{layerName}.{weightName}.scales"];
        var scales = scalesTensor.ReadBytes(originalScalesBytes);


        switch (format)
        {
            case Format.v0:
            {
                var q_weight_t = weightsSize * width;
                var q_zeros_t = zerosSize * width;
                var scales_t = scalesSize * width;

                qweights = RepackQData(qweights, height, width, q_weight_t);
                qzeros = RepackQData(qzeros, scalesSize, width, q_zeros_t);
                scales = RepackScales(scales, scalesSize, width, scales_t);
            }
            break;
            case Format.v1:
            {
                int orig_scales_height = zerosSize * 8;
                ushort[] scales_t = new ushort[scalesSize * width];
                for (int x = 0; x < width; x++)
                    for (int y = 0; y < scalesSize; y++)
                        scales_t[x * scalesSize + y] = scales[x * orig_scales_height + y];

                scales = MemoryMarshal.Cast<ushort, byte>(scales_t).ToArray();
            }
            break;
            default:
                throw new FormatException("Format is not supported");
        }

        using var writer = new BinaryWriter(outputStream, Encoding.UTF8, true);
        writer.Write(qweights);
        writer.Write(qzeros);
        writer.Write(scales);
    }

    private static byte[] RepackQData(byte[] q_weight_in_bytes, int height, int width, int size)
    {
        var q_weight_in = MemoryMarshal.Cast<byte, uint>(q_weight_in_bytes);
        var packed = new uint[(width * height) + 4];
        var q = new uint[8];

        // 1. convert to uint32 col-major array first (only 4 LSBs of each element are non-zero)
        for (var y = 0; y < height; ++y)
            for (var x = 0; x < width; x += 8)
            {
                var offset = height * x + y;
                uint packed_q_wt = q_weight_in[(y * width + x) / 8];
                for (var i = 0; i < 8; ++i)
                {
                    q[orderMap[i]] = packed_q_wt & 0xF;
                    packed_q_wt >>= 4;
                }
                for (var i = 0; i < 8; ++i)
                    packed[offset + height * i] = q[i];   // note - transpose here
            }

        var q_weight_out = new uint[size];
        // 2. pack 8 consecutive elements to single uint32_t (consecutive in the inner-most dimension which is the column dimension)
        var packed_wt_height = CeilDiv(height, 8);
        for (var x = 0; x < width; ++x)
            for (var y = 0; y < height; y += 8)
            {
                var offset = height * x + y;
                uint packed_val = 0;
                for (int i = 0; i < 8; ++i)
                    packed_val |= (packed[offset + i] << (4 * i));

                q_weight_out[x * packed_wt_height + y / 8] = packed_val;
            }

        var packed_bytes = MemoryMarshal.Cast<uint, byte>(q_weight_out);
        return packed_bytes.ToArray();
    }

    private static byte[] RepackScales(byte[] scales_in_bytes, int meta_height, int width, int size)
    {
        var scales_in = MemoryMarshal.Cast<byte, ushort>(scales_in_bytes);
        var packed = new ushort[size];
        for (var x = 0; x < width; ++x)
            for (var y = 0; y < meta_height; ++y)
                packed[x * meta_height + y] = scales_in[y * width + x];
        var packed_bytes = MemoryMarshal.Cast<ushort, byte>(packed);
        return packed_bytes.ToArray();
    }

    private class TorchUnpickler : Unpickler
    {
        static TorchUnpickler()
        {
            registerConstructor("torch._utils", "_rebuild_tensor", new TensorConstructor());
            registerConstructor("torch._utils", "_rebuild_tensor_v2", new TensorConstructor());
            registerConstructor("collections", "OrderedDict", new OrderedDictConstructor());
        }

        private readonly ZipArchive archive;

        public TorchUnpickler(ZipArchive archive) =>
            this.archive = archive;

        protected override object persistentLoad(object persistentId)
        {
            var id = (object[])persistentId;

            if ((string)id[0] != "storage")
                throw new NotImplementedException("Unknown persistent id loaded");

            var archiveKey = (string)id[2];
            return archive.Entries.First(f => f.FullName.EndsWith($"data/{archiveKey}"));
        }

        private class TensorConstructor : IObjectConstructor
        {
            public object construct(object[] args) =>
                new Tensor((ZipArchiveEntry)args[0]);
        }

        private class OrderedDict : Hashtable
        {
            public void __setstate__(Hashtable source)
            {
                foreach (DictionaryEntry entry in source)
                    this.Add(entry.Key, entry.Value);
            }
        }

        private class OrderedDictConstructor : IObjectConstructor
        {
            public object construct(object[] args) =>
                new OrderedDict();
        }
    }

    private class Tensor
    {
        private readonly ZipArchiveEntry entry;

        public Tensor(ZipArchiveEntry entry) =>
            this.entry = entry;

        public void CopyTo(Stream destination) =>
            Stream.CopyTo(destination);

        public byte[] ReadBytes(int length)
        {
            using var reader = new BinaryReader(Stream);
            return reader.ReadBytes(length);
        }

        private Stream Stream { get => entry.Open(); }
    }
}
