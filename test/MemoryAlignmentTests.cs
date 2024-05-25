namespace libLlama2.UnitTests;

using ManagedCuda;

public class MemoryAlignmentTests : IDisposable
{
    private readonly CudaContext context;

    public MemoryAlignmentTests() =>
        context = new CudaContext(deviceId: 0);

    public void Dispose() =>
        context.Dispose();

    private void MemoryAlignment<T>(string kernel_name, int data_size, int offset) where T : struct
    {
        var kernel = context.LoadKernel("vecadd_kernel.ptx", kernel_name);
        var pointer = new CudaDeviceVariable<T>(data_size);
        kernel.Run(pointer.DevicePointer, pointer.DevicePointer + (offset * pointer.TypeSize), offset);
    }

    public static IEnumerable<object[]> TestData()
    {
        const int data_size = 32;
        for (int i = 0; i < data_size; ++i)
            yield return new object[] { data_size, i };
    }

    [Theory]
    [MemberData(nameof(TestData))]
    public void Test_MemoryAlignment(int data_size, int offset) =>
        MemoryAlignment<int>("memory_alignment_int_kernel", data_size, offset);

    [Theory]
    [MemberData(nameof(TestData))]
    public void Test_MemoryAlignment_half(int data_size, int offset) =>
        MemoryAlignment<Half>("memory_alignment_half_kernel", data_size, offset);


}
