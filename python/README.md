Python Server for External Models
=================================

This folder contains code that enables running some of the leading general-purpose or math-specific models. It is also fairly easy to adapt the existing code and run other external models you would like to bring.

## Requirements

The setup steps are pretty simple. The script below is sufficient to run all external models already supported in this folder. If you only want to run a subset of them, you may not need all packages in the last step of pip installation.

### For NVIDIA CUDA:
```bash
conda create --name lean-copilot python=3.10 python numpy
conda activate lean-copilot
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Depending on your CUDA version; see https://pytorch.org/
pip install fastapi uvicorn loguru transformers openai anthropic google.generativeai vllm
```

### For AMD ROCm:
```bash
conda create --name lean-copilot python=3.10 python numpy
conda activate lean-copilot
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0  # See https://pytorch.org/get-started/locally/ for ROCm versions
pip install fastapi uvicorn loguru transformers openai anthropic google.generativeai
# Note: vllm may not support ROCm - check vllm documentation for ROCm compatibility
```

### CPU Only:
```bash
conda create --name lean-copilot python=3.10 python numpy
conda activate lean-copilot
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn loguru transformers openai anthropic google.generativeai
```

## Running the Server

```bash
uvicorn server:app --port 23337
```

After the server is up running, you can go to `LeanCopilotTests/ModelAPIs.lean` to try your external models out!

## Testing ROCm Support

To test ROCm/HIP support for AMD GPUs:

```bash
# Run the ROCm test script
python test_rocm.py
```

This will test:
- ROCm device detection
- Model loading on ROCm devices  
- Basic inference functionality

The script will automatically fall back to CPU if ROCm is not available.

## Contributions

We welcome contributions. If you think it would beneficial to add some other external models, or if you would like to make other contributions regarding the external model support in Lean Copilot, please feel free to open a PR. The main entry point is this `python` folder as well as the `ModelAPIs.lean` file under `LeanCopilotTests`.

We use [`black`](https://pypi.org/project/black/) to format code in this folder.
