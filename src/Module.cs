using ManagedCuda;

namespace libLlama2;

public abstract class Module
{
    protected readonly CudaKernel kernel;

    public Module(CudaContext cudaContext, string modulePath, string moduleName)
    {
        kernel = cudaContext.LoadKernel(modulePath, moduleName);
    }

    protected static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;
}