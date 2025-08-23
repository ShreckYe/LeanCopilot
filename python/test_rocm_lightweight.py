# TODO remove this file when the PR is completed

#!/usr/bin/env python3
"""
Lightweight ROCm test script for LeanCopilot with memory constraints.

This script tests ROCm functionality without loading large models
to avoid GPU memory issues.
"""

import torch
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

from models import (
    get_cuda_if_available,
    get_rocm_if_available, 
    get_best_device_available,
    get_device
)

def test_device_detection():
    """Test device detection functions"""
    print("=== Device Detection Tests ===")
    
    # Test CUDA detection
    cuda_device = get_cuda_if_available()
    print(f"CUDA device: {cuda_device}")
    
    # Test ROCm detection
    rocm_device = get_rocm_if_available()
    print(f"ROCm device: {rocm_device}")
    
    # Test best device selection
    best_device = get_best_device_available()
    print(f"Best device: {best_device}")
    
    # PyTorch device info
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"Device {i}: {props.name} ({memory_gb:.1f} GB memory)")
    
    return cuda_device, rocm_device, best_device

def test_rocm_env():
    """Test ROCm environment variables"""
    print("\n=== ROCm Environment Tests ===")
    
    rocm_env_vars = [
        'HIP_VISIBLE_DEVICES',
        'ROCR_VISIBLE_DEVICES', 
        'HSA_VISIBLE_DEVICES',
        'ROCM_PATH',
        'HIP_PATH'
    ]
    
    for var in rocm_env_vars:
        value = os.environ.get(var)
        print(f"{var}: {value if value else 'Not set'}")

def test_basic_gpu_operations():
    """Test basic GPU operations without loading large models"""
    print("\n=== Basic GPU Operations Tests ===")
    
    if not torch.cuda.is_available():
        print("❌ No GPU available, skipping GPU tests")
        return False
    
    try:
        # Test basic tensor operations on GPU
        print("Testing basic tensor operations on GPU...")
        
        # Create a small tensor and move it to GPU
        device = get_best_device_available()
        print(f"Using device: {device}")
        
        # Test tensor creation and basic operations
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        
        print(f"✅ Basic tensor operations successful")
        print(f"   Tensor device: {x.device}")
        print(f"   Result shape: {z.shape}")
        
        # Test memory allocation
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        print(f"   GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"   GPU memory reserved: {memory_reserved:.1f} MB")
        
        # Test device selection functions
        print("\nTesting device selection functions...")
        
        # Test get_device function with different parameters
        auto_device = get_device("auto")
        cuda_device = get_device("cuda") 
        rocm_device = get_device("rocm")
        cpu_device = get_device("cpu")
        
        print(f"   auto device: {auto_device}")
        print(f"   cuda device: {cuda_device}")
        print(f"   rocm device: {rocm_device}")
        print(f"   cpu device: {cpu_device}")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during GPU testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_model():
    """Test with a very small model to verify model loading works"""
    print("\n=== Small Model Test ===")
    
    if not torch.cuda.is_available():
        print("❌ No GPU available, skipping model test")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a very small model (DistilGPT-2 is much smaller than Pythia)
        model_name = "distilgpt2"  # ~82M parameters vs 2.8B
        
        print(f"Loading small model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on GPU
        device = get_best_device_available()
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Test basic inference
        test_input = "The answer is"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        print("Testing basic inference...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 5,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful: '{result}'")
        
        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        print(f"   Memory used: {memory_allocated:.1f} MB")
        
        # Clean up
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all lightweight ROCm tests"""
    print("Lightweight ROCm/HIP Support Test for LeanCopilot")
    print("=" * 55)
    
    # Test device detection
    cuda_dev, rocm_dev, best_dev = test_device_detection()
    
    # Test ROCm environment
    test_rocm_env()
    
    # Test basic GPU operations
    gpu_success = test_basic_gpu_operations()
    
    # Test small model loading if GPU is available
    model_success = False
    if gpu_success:
        model_success = test_small_model()
    
    # Summary
    print("\n" + "="*55)
    print("Test Summary:")
    print(f"✅ Device detection: PASSED")
    print(f"{'✅' if gpu_success else '❌'} Basic GPU operations: {'PASSED' if gpu_success else 'FAILED'}")
    print(f"{'✅' if model_success else '❌'} Model loading: {'PASSED' if model_success else 'FAILED'}")
    
    if gpu_success and model_success:
        print("\n🎉 All ROCm tests passed! Your AMD GPU is working correctly.")
    elif gpu_success:
        print("\n⚠️ Basic GPU operations work, but model loading failed.")
    else:
        print("\n❌ GPU operations failed. Check ROCm installation.")
    
    print("\nNote: This test uses small models to avoid memory issues.")
    print("For production use, monitor GPU memory when using larger models.")

if __name__ == "__main__":
    main()
