using System.Reflection;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace libLlama2;

public abstract class Module
{
    protected readonly CudaKernel kernel;

    private static readonly Assembly assembly = Assembly.GetExecutingAssembly();

    private static readonly CUstream stream = CUstream.NullStream;

    public Module(CudaContext cudaContext, string moduleFile, string moduleName)
    {
        using var moduleStream = assembly.GetManifestResourceStream(moduleFile);
        kernel = cudaContext.LoadKernelPTX(moduleStream, moduleName);
    }

    protected void Forward(params object[] parameters) =>
        kernel.RunAsync(stream, parameters);

    protected static int CeilDiv(int a, int b) =>
        (a + (b - 1)) / b;
}