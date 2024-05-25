## ManagedLlama2

Llama2 inference for 4-bit AWQ quantized models with C# and [ManagedCuda](https://github.com/kunzmi/managedCuda)

Based on [llama2.c](https://github.com/karpathy/llama2.c) and [llama_cu_awq](https://github.com/ankan-ban/llama_cu_awq)

## Build

```
git clone https://github.com/GilesBathgate/ManagedLlama2
cd ManagedLlama2
./build.sh
```

## Setup

Easily download and convert the model

```
cd examples/Setup
dotnet run ../model.bin ../tokenizer.bin
cd -
```

## Run

Launch a simple web based chat client

```
cd examples/WebSocket
dotnet run ../model.bin ../tokenizer.bin
cd -
```

## License

MIT / GPLv3