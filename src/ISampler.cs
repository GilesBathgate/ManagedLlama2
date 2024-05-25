namespace libLlama2;

public interface ISampler
{
    public int Sample(int nextPosition, bool generateToken);

}