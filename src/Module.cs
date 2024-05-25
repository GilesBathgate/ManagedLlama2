using ManagedCuda;

namespace libLlama2;

public abstract class Module
{
    protected readonly CudaKernel kernel;

    private static readonly string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

    public Module(CudaContext cudaContext, string moduleFile, string moduleName)
    {
        var modulePath = Path.Combine(baseDirectory, moduleFile);
        kernel = cudaContext.LoadKernel(modulePath, moduleName);
    }

    protected static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;
}