using System.Collections.Generic;

namespace libLlama2;

public interface IDynamicConstraintStateMachine : IConstraintStateMachine
{
    (bool allowed, List<int> tokenIds) GetActiveTokens(ITokenizer tokenizer, int vocabSize);
}
