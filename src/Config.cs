using System.Runtime.InteropServices;

namespace libLlama2;

[StructLayout(LayoutKind.Sequential)]
public struct Config
{
    public int dim; // transformer dimension
    public int hidden_dim; // for ffn layers
    public int n_layers; // number of layers
    public int n_heads; // number of query heads
    public int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    public int vocab_size; // vocabulary size, usually 256 (byte-level)
    public int seq_len; // max sequence length
    public float rope_theta; // theta for the rope rotational embedding
    public static int Size { get => Marshal.SizeOf(typeof(Config)); }
}