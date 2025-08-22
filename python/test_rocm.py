#!/usr/bin/env python3
"""
Test script for ROCm/HIP support in LeanCopilot.

This script tests:
1. ROCm device detection
2. Model loading on ROCm devices
3. Basic inference with ROCm

Run this on a system with ROCm-enabled PyTorch to test AMD GPU support.
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
    PythiaTacticGenerator
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
            print(f"Device {i}: {props.name} (Compute capability: {props.major}.{props.minor})")
    
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

def test_model_on_rocm():
    """Test loading a small model on ROCm device"""
    print("\n=== Model Loading Tests ===")
    
    try:
        # Test with auto device selection
        print("Testing with auto device selection...")
        model_auto = PythiaTacticGenerator(
            num_return_sequences=1, 
            max_length=64,
            device="auto"
        )
        print(f"Model loaded on device: {model_auto.device}")
        
        # Test explicit ROCm device
        print("Testing with explicit rocm device selection...")
        model_rocm = PythiaTacticGenerator(
            num_return_sequences=1,
            max_length=64, 
            device="rocm"
        )
        print(f"ROCm model loaded on device: {model_rocm.device}")
        
        # Test basic inference
        test_input = "n : ℕ\n⊢ gcd n n = n"
        print(f"Testing inference with input: {test_input}")
        
        result = model_rocm.generate(test_input)
        print(f"Generated result: {result[:2] if len(result) > 2 else result}")  # Show first 2 results
        
        return True
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ROCm tests"""
    print("ROCm/HIP Support Test for LeanCopilot")
    print("=" * 50)
    
    # Test device detection
    cuda_dev, rocm_dev, best_dev = test_device_detection()
    
    # Test ROCm environment
    test_rocm_env()
    
    # Test model loading (only if we have GPU available)
    if torch.cuda.is_available():
        success = test_model_on_rocm()
        if success:
            print("\n✅ All ROCm tests passed!")
        else:
            print("\n❌ Some ROCm tests failed!")
    else:
        print("\n⚠️ No GPU available, skipping model tests")
    
    print("\nNote: For full ROCm testing, run this on a system with:")
    print("- AMD GPU with ROCm support") 
    print("- PyTorch compiled with ROCm support")
    print("- ROCm drivers and runtime installed")

if __name__ == "__main__":
    main()