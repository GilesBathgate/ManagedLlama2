namespace libLlama2.UnitTests;

using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda;
using Xunit;

public class ConstrainTests : IDisposable
{
    private readonly CudaContext cudaContext;
    private readonly CudaKernel kernel;

    public ConstrainTests()
    {
        int deviceID = 0;
        cudaContext = new CudaContext(deviceID);
        kernel = cudaContext.LoadKernel("sample_constrained_kernel.ptx", "sample_constrained_kernel");
        kernel.GridDimensions = 1;
        kernel.BlockDimensions = 1024;
    }

    public void Dispose() =>
        cudaContext.Dispose();

    private Half[] Constrain(Half[] x, int[] constraint)
    {
        var size = x.Length;
        var x1 = (CudaDeviceVariable<Half>)x;
        var c1 = (CudaDeviceVariable<int>)constraint;

        kernel.Run(x1.DevicePointer, size, c1.DevicePointer, c1.Size);
        return (Half[])x1;
    }

    private Half[] Expected(Half[] logits, int[] constraint)
    {
        var size = logits.Length;
        var constraintSet = new HashSet<int>(constraint);

        for (var i = 0; i < size; ++i)
        {
            if (!constraintSet.Contains(i)) {
                logits[i] = (Half)float.NegativeInfinity;
            }
        }

        return logits;
    }

    private static Half[] Copy(Half[] x)
    {
        var copy_x = new Half[x.Length];
        x.CopyTo(copy_x, 0);
        return copy_x;
    }

    public static IEnumerable<object[]> RandomData()
    {
        var generator = new Generator();
        var random = new Random();
        const int count = 10;
        for (int i = 0; i < count; ++i)
        {
            var vocabSize = random.Next(10, 20);
            var constraintSize = random.Next(1, vocabSize);
            var constraint = generator.NextIntArray(constraintSize, 0, vocabSize).ToArray();
            yield return new object[] { generator.NextArray(vocabSize), constraint };
        }
        yield return new object[] { generator.NextArray(16), new int[] { 0 } };
        yield return new object[] { generator.NextArray(16), Enumerable.Range(0,16).ToArray() };
    }

    [Theory]
    [MemberData(nameof(RandomData))]
    public void Test_Constrain(Half[] x, int[] constraint)
    {
        var actual_x = Copy(x);
        var actual = Constrain(actual_x, constraint);

        var expected_x = Copy(x);
        var expected = Expected(expected_x, constraint);

        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }
}
