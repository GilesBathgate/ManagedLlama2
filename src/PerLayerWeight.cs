using ManagedCuda;

namespace libLlama2;

public class PerLayerWeight
{
    public required CudaDeviceVariable<Half> rmsAttentionWeight;
    public required CudaDeviceVariable<Half> rmsFeedForwardWeight;
    public required QWeight queryWeight;
    public required QWeight keyWeight;
    public required QWeight valueWeight;
    public required QWeight outputWeight;
    public required QWeight gateWeight;
    public required QWeight upWeight;
    public required QWeight downWeight;
}