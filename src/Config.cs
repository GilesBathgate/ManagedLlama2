using System.Runtime.InteropServices;
using System.Text;

namespace libLlama2;

[StructLayout(LayoutKind.Sequential)]
public struct Config
{
    public int dim; // transformer dimension
    public int hiddenDim; // for ffn layers
    public int numLayers; // number of layers
    public int numHeads; // number of query heads
    public int numKVHeads; // number of key/value heads (can be < query heads because of multiquery)
    public int vocabSize; // vocabulary size, usually 256 (byte-level)
    public int seqLength; // max sequence length
    public float ropeTheta; // theta for the rope rotational embedding

    // The following are static which ensures the struct layout is preserved
    public static int Size { get => Marshal.SizeOf(typeof(Config)); }

    public static Config LoadConfig(FileStream fileStream)
    {
        using var reader = new BinaryReader(fileStream, Encoding.UTF8, true);
        var bytes = reader.ReadBytes(Size);
        var config = MemoryMarshal.Cast<byte, Config>(bytes)[0];

        if (config.ropeTheta != 10000.0f)
            throw new FileLoadException("Invalid model config");

        return config;
    }

    public static void SaveConfig(Stream stream, Config config)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, true);
        var bytes = MemoryMarshal.Cast<Config, byte>(new[] { config });
        writer.Write(bytes);
    }
}