using System.Reflection;
using ManagedCuda;

namespace libLlama2;

public abstract class Module
{
    protected readonly CudaKernel kernel;

    private static readonly Assembly assembly = Assembly.GetExecutingAssembly();

    public Module(CudaContext cudaContext, string moduleFile, string moduleName)
    {
        using var stream = assembly.GetManifestResourceStream(moduleFile);
        kernel = cudaContext.LoadKernelPTX(stream, moduleName);
    }

    protected static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;
}