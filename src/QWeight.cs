namespace libLlama2;

using ManagedCuda;

public class QWeight
{
    public required CudaDeviceVariable<uint> Weight;
    public required CudaDeviceVariable<uint> Zeros;
    public required CudaDeviceVariable<Half> Scales;
}

public class HostQWeight
{
    public uint[] Weight;

    public uint[] Zeros;

    public Half[] Scales;

    public HostQWeight(int size) =>
        (Weight, Zeros, Scales) = (new uint[size], new uint[size / 128], new Half[size / 16]);

    private const int iSize = 8; // uint32 has 8 4bit ints;
    private const int wSize = 4;
    private const int zSize = 32;
    private const int nBits = 4;
    private const uint mask = 0xF;

    public Half[] Unpack()
    {
        var output = new float[Weight.Length * iSize];

        var xSize = Weight.Length / wSize;
        for (var x = 0; x < xSize; ++x)
        {
            var zIndex = x / zSize;
            var sIndex = x / wSize;
            var wIndex = x * wSize;
            var zero = (int)((Zeros[zIndex] >> (sIndex * nBits)) & mask);
            var scale = Scales[sIndex];
            for (var w = 0; w < wSize; ++w)
            {
                var weight = Weight[wIndex + w];
                var oIndex = wIndex * iSize + w * iSize;
                for (var i = 0; i < iSize; ++i)
                {
                    var element = (int)((weight >> (i * nBits)) & mask);
                    output[oIndex + i] = (element - zero) * (float)scale;
                }
            }
        }

        return output.ToHalf();
    }

    public static HostQWeight Pack(Half[] data, float[] scales, uint[] zeros)
    {
        var q = new HostQWeight(data.Length / 8);

        for (var i = 0; i < zeros.Length / iSize; ++i)
        {
            uint combined = 0;
            var zIndex = i * iSize;
            for (var z = 0; z < iSize; ++z)
                combined |= (zeros[zIndex + z] & mask) << (z * nBits);
            q.Zeros[i] = combined;
        }

        var sSize = scales.Length * wSize;
        for (var s = 0; s < sSize; ++s)
        {
            var sIndex = s / wSize;
            var wIndex = s * wSize;
            var scale = scales[sIndex];
            var zero = zeros[sIndex];
            for (var w = 0; w < wSize; ++w)
            {
                uint weight = 0;
                var oIndex = wIndex * iSize + w * iSize;
                for (var i = 0; i < iSize; ++i)
                {
                    var element = (uint)(((float)data[oIndex + i] / scale) + zero);
                    weight |= (element & mask) << (i * nBits);
                }
                q.Weight[wIndex + w] = weight;
            }
            q.Scales[sIndex] = (Half)scale;
        }


        return q;
    }
}