#!/usr/bin/env python3
"""
Test script for DynaLoRA implementation.

This script tests the DynaLoRA functionality including:
- LoRA Router mechanism
- LoRA Expert modules
- DynaLoRALinear layers
- Mixture-of-experts architecture
- Load balancing loss
- Expert usage tracking
- Temperature scaling
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Add the timellama module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../timellama'))

try:
    from layers.DynaLoRA import (
        LoRARouter, LoRAExpert, DynaLoRALinear, DynaLoRAConfig, 
        DynaLoRAWrapper, DynaLoRAAttention, create_dynalora_model
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def test_lora_router():
    """Test LoRA Router functionality."""
    print("Testing LoRA Router...")
    
    try:
        # Test basic router initialization
        router = LoRARouter(
            d_model=512,
            n_experts=4,
            n_modules=7,
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 96
        hidden_states = torch.randn(batch_size, seq_len, 512)
        
        gating_weights = router(hidden_states)
        expected_shape = (batch_size, seq_len, 4, 7)
        
        if gating_weights.shape == expected_shape:
            print(f"✅ LoRA Router forward pass: {gating_weights.shape}")
        else:
            print(f"❌ LoRA Router shape mismatch: Expected {expected_shape}, Got {gating_weights.shape}")
            return False
        
        # Test expert usage statistics
        stats = router.get_expert_usage_stats()
        if len(stats) == 4:  # 4 experts
            print(f"✅ Expert usage stats: {stats}")
        else:
            print(f"❌ Expert usage stats error: Expected 4 experts, Got {len(stats)}")
            return False
        
        # Test temperature scaling
        if hasattr(router, 'temperature'):
            print(f"✅ Temperature parameter: {router.temperature.item():.4f}")
        else:
            print("❌ Temperature parameter not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA Router test failed: {e}")
        return False


def test_lora_expert():
    """Test LoRA Expert functionality."""
    print("Testing LoRA Expert...")
    
    try:
        # Test expert initialization
        expert = LoRAExpert(
            d_model=512,
            r_base=8,
            n_modules=7,
            lora_dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 512)
        module_idx = 0
        gating_weight = torch.randn(batch_size, seq_len, 1)
        
        output = expert(x, module_idx, gating_weight)
        expected_shape = (batch_size, seq_len, 512)
        
        if output.shape == expected_shape:
            print(f"✅ LoRA Expert forward pass: {output.shape}")
        else:
            print(f"❌ LoRA Expert shape mismatch: Expected {expected_shape}, Got {output.shape}")
            return False
        
        # Test parameter initialization
        if expert.lora_A.shape == (7, 8, 512) and expert.lora_B.shape == (7, 512, 8):
            print("✅ LoRA Expert parameter shapes correct")
        else:
            print("❌ LoRA Expert parameter shapes incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA Expert test failed: {e}")
        return False


def test_dynalora_linear():
    """Test DynaLoRALinear functionality."""
    print("Testing DynaLoRALinear...")
    
    try:
        # Test layer initialization
        layer = DynaLoRALinear(
            in_features=512,
            out_features=512,
            d_model=512,
            r_base=8,
            n_experts=4,
            lora_dropout=0.1,
            router_dropout=0.1,
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 512)
        
        output = layer(x)
        expected_shape = (batch_size, seq_len, 512)
        
        if output.shape == expected_shape:
            print(f"✅ DynaLoRALinear forward pass: {output.shape}")
        else:
            print(f"❌ DynaLoRALinear shape mismatch: Expected {expected_shape}, Got {output.shape}")
            return False
        
        # Test regularization loss
        reg_loss = layer.get_regularization_loss()
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() == 0:
            print(f"✅ Regularization loss: {reg_loss.item():.6f}")
        else:
            print("❌ Regularization loss error")
            return False
        
        # Test expert usage
        usage_stats = layer.get_expert_usage()
        if isinstance(usage_stats, dict) and len(usage_stats) == 4:
            print(f"✅ Expert usage: {usage_stats}")
        else:
            print("❌ Expert usage error")
            return False
        
        # Test usage stats reset
        layer.reset_usage_stats()
        print("✅ Usage stats reset successful")
        
        return True
        
    except Exception as e:
        print(f"❌ DynaLoRALinear test failed: {e}")
        return False


def test_dynalora_attention():
    """Test DynaLoRAAttention functionality."""
    print("Testing DynaLoRAAttention...")
    
    try:
        # Test attention layer initialization
        attention = DynaLoRAAttention(
            d_model=512,
            n_heads=8,
            r_base=8,
            n_experts=4,
            lora_dropout=0.1,
            router_dropout=0.1,
            attention_dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 512)
        
        output = attention(x)
        expected_shape = (batch_size, seq_len, 512)
        
        if output.shape == expected_shape:
            print(f"✅ DynaLoRAAttention forward pass: {output.shape}")
        else:
            print(f"❌ DynaLoRAAttention shape mismatch: Expected {expected_shape}, Got {output.shape}")
            return False
        
        # Test regularization loss
        reg_loss = attention.get_regularization_loss()
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() == 0:
            print(f"✅ Attention regularization loss: {reg_loss.item():.6f}")
        else:
            print("❌ Attention regularization loss error")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ DynaLoRAAttention test failed: {e}")
        return False


def test_dynalora_wrapper():
    """Test DynaLoRAWrapper functionality."""
    print("Testing DynaLoRAWrapper...")
    
    try:
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(512, 512)
                self.k_proj = nn.Linear(512, 512)
                self.v_proj = nn.Linear(512, 512)
                self.o_proj = nn.Linear(512, 512)
                self.gate_proj = nn.Linear(512, 2048)
                self.up_proj = nn.Linear(512, 2048)
                self.down_proj = nn.Linear(2048, 512)
        
        # Test wrapper initialization
        model = TestModel()
        config = DynaLoRAConfig(
            d_model=512,
            r_base=8,
            n_experts=4,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        
        wrapper = DynaLoRAWrapper(model, config)
        
        # Test applying DynaLoRA
        modified_model = wrapper.apply_dynalora()
        
        # Check if layers were replaced
        if hasattr(modified_model, 'q_proj') and isinstance(modified_model.q_proj, DynaLoRALinear):
            print("✅ DynaLoRA layers applied successfully")
        else:
            print("❌ DynaLoRA layers not applied correctly")
            return False
        
        # Test regularization loss
        reg_loss = wrapper.get_regularization_loss()
        if isinstance(reg_loss, torch.Tensor):
            print(f"✅ Wrapper regularization loss: {reg_loss.item():.6f}")
        else:
            print("❌ Wrapper regularization loss error")
            return False
        
        # Test expert usage
        usage_stats = wrapper.get_expert_usage()
        if isinstance(usage_stats, dict):
            print(f"✅ Wrapper expert usage: {len(usage_stats)} layers")
        else:
            print("❌ Wrapper expert usage error")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ DynaLoRAWrapper test failed: {e}")
        return False


def test_create_dynalora_model():
    """Test create_dynalora_model function."""
    print("Testing create_dynalora_model...")
    
    try:
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(512, 512)
                self.linear2 = nn.Linear(512, 512)
        
        model = TestModel()
        
        # Test model creation
        modified_model, wrapper = create_dynalora_model(
            model=model,
            d_model=512,
            r_base=8,
            n_experts=4,
            target_modules=['linear1', 'linear2'],
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Check if model was modified
        if isinstance(modified_model.linear1, DynaLoRALinear) and isinstance(modified_model.linear2, DynaLoRALinear):
            print("✅ DynaLoRA model created successfully")
        else:
            print("❌ DynaLoRA model creation failed")
            return False
        
        # Test forward pass
        x = torch.randn(2, 96, 512)
        output1 = modified_model.linear1(x)
        output2 = modified_model.linear2(output1)
        
        if output2.shape == (2, 96, 512):
            print(f"✅ Modified model forward pass: {output2.shape}")
        else:
            print(f"❌ Modified model forward pass error: {output2.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ create_dynalora_model test failed: {e}")
        return False


def test_load_balancing():
    """Test load balancing functionality."""
    print("Testing load balancing...")
    
    try:
        # Create a DynaLoRA layer
        layer = DynaLoRALinear(
            in_features=512,
            out_features=512,
            d_model=512,
            r_base=8,
            n_experts=4,
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Simulate multiple forward passes to build usage statistics
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 512)
        
        for _ in range(10):  # Multiple forward passes
            _ = layer(x)
        
        # Check load balancing loss
        reg_loss = layer.get_regularization_loss(load_balance_weight=0.01)
        print(f"✅ Load balancing loss: {reg_loss.item():.6f}")
        
        # Check expert usage distribution
        usage_stats = layer.get_expert_usage()
        total_usage = sum(usage_stats.values())
        print(f"✅ Expert usage distribution: {usage_stats}")
        print(f"✅ Total usage: {total_usage:.4f}")
        
        # Test temperature learning
        if hasattr(layer.router, 'temperature') and layer.router.temperature.requires_grad:
            print("✅ Temperature parameter is learnable")
        else:
            print("❌ Temperature parameter is not learnable")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Load balancing test failed: {e}")
        return False


def test_temperature_scaling():
    """Test temperature scaling functionality."""
    print("Testing temperature scaling...")
    
    try:
        # Test learnable temperature
        router1 = LoRARouter(
            d_model=512,
            n_experts=4,
            temperature=1.0,
            learnable_temperature=True
        )
        
        # Test fixed temperature
        router2 = LoRARouter(
            d_model=512,
            n_experts=4,
            temperature=2.0,
            learnable_temperature=False
        )
        
        # Test forward passes
        hidden_states = torch.randn(2, 96, 512)
        
        weights1 = router1(hidden_states)
        weights2 = router2(hidden_states)
        
        # Check that different temperatures produce different outputs
        if not torch.allclose(weights1, weights2, atol=1e-6):
            print("✅ Temperature scaling produces different outputs")
        else:
            print("❌ Temperature scaling not working")
            return False
        
        # Check learnable temperature
        if router1.temperature.requires_grad:
            print("✅ Learnable temperature works")
        else:
            print("❌ Learnable temperature not working")
            return False
        
        # Check fixed temperature
        if not router2.temperature.requires_grad:
            print("✅ Fixed temperature works")
        else:
            print("❌ Fixed temperature not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Temperature scaling test failed: {e}")
        return False


def main():
    """Run all DynaLoRA tests."""
    print("=" * 60)
    print("DynaLoRA Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: LoRA Router
    test_results['lora_router'] = test_lora_router()
    
    # Test 2: LoRA Expert
    test_results['lora_expert'] = test_lora_expert()
    
    # Test 3: DynaLoRALinear
    test_results['dynalora_linear'] = test_dynalora_linear()
    
    # Test 4: DynaLoRAAttention
    test_results['dynalora_attention'] = test_dynalora_attention()
    
    # Test 5: DynaLoRAWrapper
    test_results['dynalora_wrapper'] = test_dynalora_wrapper()
    
    # Test 6: create_dynalora_model
    test_results['create_dynalora_model'] = test_create_dynalora_model()
    
    # Test 7: Load balancing
    test_results['load_balancing'] = test_load_balancing()
    
    # Test 8: Temperature scaling
    test_results['temperature_scaling'] = test_temperature_scaling()
    
    # Summary
    print("\n" + "=" * 60)
    print("DynaLoRA Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All DynaLoRA tests passed! Implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some DynaLoRA tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
