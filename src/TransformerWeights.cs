using ManagedCuda;

namespace libLlama2;

public class TransformerWeights
{
    public required CudaDeviceVariable<Half> tokenEmbeddingTable; // token embedding table
    public required CudaDeviceVariable<Half> classifierWeights; // classifier weights for the logits, on the last layer
    public required CudaDeviceVariable<Half> rmsFinalWeight; // final rmsnorm
    public required ICollection<PerLayerWeight> layers;
}