# ROCm/HIP Support Implementation for LeanCopilot

This document describes the ROCm/HIP support implementation for LeanCopilot and provides instructions for testing.

## What Was Implemented

### 1. Device Type Support
- Added `rocm` device type to the `Device` enum in `LeanCopilot/Models/Native.lean`
- Extended device string conversion to include ROCm

### 2. FFI (Foreign Function Interface)
- Added `rocm_available()` C++ function in `cpp/ct2.cpp`
- Added `rocmAvailable` Lean function in `LeanCopilot/Models/FFI.lean`
- ROCm detection uses environment variable checks (`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`)

### 3. Python Backend Support
- Added `get_rocm_if_available()` function that detects AMD GPUs through PyTorch
- Added `get_best_device_available()` function for automatic device selection
- Updated all model classes to support `device="rocm"` parameter
- Added `rocm()` and `to_best_device()` methods to Transformer classes

### 4. Build System
- Added `hasROCm()` and `useROCm()` functions to lakefile.lean
- Updated build archive naming to include ROCm variants
- ROCm detection checks for `hipcc` and `rocm-smi` tools

### 5. Testing
- Added ROCm availability test in `LeanCopilotTests/ModelAPIs.lean`
- Created comprehensive test script `python/test_rocm.py`
- Added ROCm-specific model configuration example

### 6. Documentation
- Updated `python/README.md` with ROCm installation instructions
- Added separate sections for NVIDIA CUDA, AMD ROCm, and CPU-only setups

## Testing Instructions

### Prerequisites for ROCm Testing

1. **AMD GPU with ROCm Support**:
   - AMD Radeon RX Vega, RX 6000, RX 7000 series
   - AMD Radeon Pro WX series
   - AMD Instinct MI series

2. **ROCm Installation**:
   ```bash
   # Ubuntu/Debian
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev rocm-libs rocm-utils
   
   # Add user to render group
   sudo usermod -a -G render $USER
   ```

3. **PyTorch with ROCm**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```

### Running Tests

1. **Python ROCm Test**:
   ```bash
   cd python
   python test_rocm.py
   ```

2. **Lean ROCm Tests**:
   ```bash
   # Build the project (requires Lean 4 and Lake)
   lake build
   
   # Run tests in LeanCopilotTests/ModelAPIs.lean
   lake exe lean --run LeanCopilotTests/ModelAPIs.lean
   ```

3. **Manual Model Testing**:
   ```python
   from models import PythiaTacticGenerator
   
   # Test with explicit ROCm device
   model = PythiaTacticGenerator(
       num_return_sequences=4,
       max_length=256,
       device="rocm"
   )
   
   result = model.generate("n : ℕ\n⊢ gcd n n = n")
   print(result)
   ```

### Expected Behavior

- **With ROCm**: Models load on AMD GPU, device shows as `cuda:0` (ROCm uses CUDA API)
- **Without ROCm**: Automatically falls back to CPU
- **Device Detection**: `rocmAvailable` returns `true` if ROCm environment is detected

## Fallback Behavior

The implementation is designed to be safe and fallback gracefully:

1. If ROCm is requested but not available → falls back to CPU
2. If `device="auto"` → automatically selects best available (CUDA > ROCm > CPU)
3. All existing CUDA functionality remains unchanged
4. CPU-only operation works exactly as before

## Implementation Notes

### ROCm vs CUDA API
- ROCm uses the same PyTorch CUDA API internally via HIP compatibility layer
- Models use `torch.cuda.*` functions even on AMD GPUs
- This is standard practice for ROCm in PyTorch

### CTranslate2 Limitation
- CTranslate2 may not have native ROCm support
- C++ implementation focuses on environment detection
- Full ROCm acceleration requires CTranslate2 ROCm backend (if available)

### Architecture Support
- Primarily tested on x86_64 Linux
- Windows ROCm support is limited
- macOS does not support ROCm (Apple Silicon uses different compute)

## Troubleshooting

1. **ROCm not detected but GPU present**:
   - Check ROCm installation: `rocm-smi`
   - Verify environment variables: `echo $HIP_VISIBLE_DEVICES`
   - Check PyTorch ROCm: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Model fails to load on ROCm**:
   - Check VRAM availability: `rocm-smi`
   - Try smaller models or reduce batch size
   - Verify PyTorch ROCm compilation: `torch.version.hip`

3. **Performance issues**:
   - ROCm performance may differ from CUDA
   - Consider model quantization for better performance
   - Monitor GPU utilization with `rocm-smi -d`

## Future Improvements

1. **Enhanced ROCm Detection**: More sophisticated GPU detection beyond environment variables
2. **CTranslate2 ROCm Backend**: If/when CTranslate2 adds native ROCm support
3. **Performance Optimization**: ROCm-specific optimizations and tuning
4. **Additional Testing**: More comprehensive test coverage for various AMD GPU architectures

This implementation provides a solid foundation for ROCm support in LeanCopilot while maintaining full backward compatibility with existing CUDA and CPU workflows.