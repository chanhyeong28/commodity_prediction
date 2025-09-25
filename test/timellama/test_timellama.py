#!/usr/bin/env python3
"""
Test script for TimeLlaMA model implementation.

This script tests the core TimeLlaMA functionality including:
- Model initialization and forward pass
- Channel-as-token embedding
- Prompt alignment
- Reprogramming layer
- LLM backbone integration
- Ablation variants
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
    from model.TimeLlaMA import TimeLlaMA
    from layers.Embed import TSEmb, TSEmbConv, TSEmbHybrid
    from layers.prompt_align import PromptAlignment
    from layers.SelfAttention_Family import CrossAttention
    from utils.metrics import metric, short_term_metrics
    from utils.losses import smape_loss
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class MockLLMModel(nn.Module):
    """Mock LLM model for testing purposes."""
    
    def __init__(self, d_model: int = 512, vocab_size: int = 32000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Mock input embeddings
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
        # Mock transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ) for _ in range(8)
        ])
        
        # Mock output
        self.last_hidden_state = None
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embeddings(input_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Mock output
        class MockOutput:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states
        
        return MockOutput(x)


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = '[PAD]'
        self.eos_token = '[EOS]'
    
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        # Mock tokenization
        batch_size = len(texts) if isinstance(texts, list) else 1
        
        # Create mock token IDs
        input_ids = torch.randint(0, self.vocab_size, (batch_size, min(max_length or 100, 100)))
        attention_mask = torch.ones_like(input_ids)
        
        if return_tensors == "pt":
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        return input_ids


def test_timellama_model_initialization():
    """Test TimeLlaMA model initialization."""
    print("Testing TimeLlaMA model initialization...")
    
    # Create mock components
    llm_model = MockLLMModel(d_model=512)
    tokenizer = MockTokenizer()
    
    try:
        # Test basic initialization
        model = TimeLlaMA(
            llm_model=llm_model,
            tokenizer=tokenizer,
            d_model=512,
            lookback=96,
            pred_len=96,
            num_channels=21,
            num_heads=8,
            d_ff=2048,
            embedding_type='linear',
            use_positional_emb=True,
            use_channel_emb=True,
            dropout=0.1,
            head_dropout=0.1,
            use_reprogramming=True,
            description="Test TimeLlaMA model",
            freeze_llm=False,
            use_dynalora=True,
            dynalora_r_base=8,
            dynalora_n_experts=4,
            dynalora_dropout=0.0,
            dynalora_router_dropout=0.1
        )
        
        print("✅ TimeLlaMA model initialization successful")
        return model
        
    except Exception as e:
        print(f"❌ TimeLlaMA model initialization failed: {e}")
        return None


def test_timellama_forward_pass(model):
    """Test TimeLlaMA forward pass."""
    print("Testing TimeLlaMA forward pass...")
    
    if model is None:
        print("❌ Cannot test forward pass - model is None")
        return False
    
    try:
        # Create test input
        batch_size = 2
        seq_len = 96
        num_channels = 21
        
        x_enc = torch.randn(batch_size, seq_len, num_channels)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
        
        # Forward pass
        with torch.no_grad():
            output = model(x_enc, x_mark_enc)
        
        # Check output shape
        expected_shape = (batch_size, num_channels, model.pred_len)
        if output.shape == expected_shape:
            print(f"✅ Forward pass successful - Output shape: {output.shape}")
            return True
        else:
            print(f"❌ Forward pass failed - Expected: {expected_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        return False


def test_ablation_variants():
    """Test ablation variants."""
    print("Testing ablation variants...")
    
    llm_model = MockLLMModel(d_model=512)
    tokenizer = MockTokenizer()
    
    variants = ['full', '1', '2', '3', '4', '5', '6']
    results = {}
    
    for variant in variants:
        try:
            model = TimeLlaMA(
                llm_model=llm_model,
                tokenizer=tokenizer,
                d_model=512,
                lookback=96,
                pred_len=96,
                num_channels=21,
                num_heads=8,
                d_ff=2048,
                embedding_type='linear',
                use_positional_emb=True,
                use_channel_emb=True,
                dropout=0.1,
                head_dropout=0.1,
                use_reprogramming=True,
                description=f"Test TimeLlaMA-{variant}",
                freeze_llm=False,
                use_dynalora=True,
                dynalora_r_base=8,
                dynalora_n_experts=4,
                dynalora_dropout=0.0,
                dynalora_router_dropout=0.1,
                ablation_variant=variant
            )
            
            # Test forward pass
            x_enc = torch.randn(1, 96, 21)
            x_mark_enc = torch.randn(1, 96, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc)
            
            results[variant] = True
            print(f"✅ TimeLlaMA-{variant} test successful")
            
        except Exception as e:
            results[variant] = False
            print(f"❌ TimeLlaMA-{variant} test failed: {e}")
    
    success_count = sum(results.values())
    print(f"✅ Ablation variants test: {success_count}/{len(variants)} successful")
    return success_count == len(variants)


def test_embedding_variants():
    """Test different embedding types."""
    print("Testing embedding variants...")
    
    llm_model = MockLLMModel(d_model=512)
    tokenizer = MockTokenizer()
    
    embedding_types = ['linear', 'conv', 'hybrid']
    results = {}
    
    for emb_type in embedding_types:
        try:
            model = TimeLlaMA(
                llm_model=llm_model,
                tokenizer=tokenizer,
                d_model=512,
                lookback=96,
                pred_len=96,
                num_channels=21,
                num_heads=8,
                d_ff=2048,
                embedding_type=emb_type,
                use_positional_emb=True,
                use_channel_emb=True,
                dropout=0.1,
                head_dropout=0.1,
                use_reprogramming=True,
                description=f"Test {emb_type} embedding",
                freeze_llm=False,
                use_dynalora=False  # Disable DynaLoRA for simpler test
            )
            
            # Test forward pass
            x_enc = torch.randn(1, 96, 21)
            x_mark_enc = torch.randn(1, 96, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc)
            
            results[emb_type] = True
            print(f"✅ {emb_type} embedding test successful")
            
        except Exception as e:
            results[emb_type] = False
            print(f"❌ {emb_type} embedding test failed: {e}")
    
    success_count = sum(results.values())
    print(f"✅ Embedding variants test: {success_count}/{len(embedding_types)} successful")
    return success_count == len(embedding_types)


def test_metrics_and_losses():
    """Test metrics and loss functions."""
    print("Testing metrics and loss functions...")
    
    try:
        # Test data
        pred = np.random.randn(100, 10)
        true = np.random.randn(100, 10)
        
        # Test long-term metrics
        mae, mse, rmse, mape, mspe = metric(pred, true)
        print(f"✅ Long-term metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
        
        # Test short-term metrics
        smape, mase, owa = short_term_metrics(pred, true)
        print(f"✅ Short-term metrics: SMAPE={smape:.4f}, MASE={mase:.4f}, OWA={owa:.4f}")
        
        # Test SMAPE loss
        pred_tensor = torch.tensor(pred, dtype=torch.float32)
        true_tensor = torch.tensor(true, dtype=torch.float32)
        
        smape_loss_fn = smape_loss()
        loss_value = smape_loss_fn(pred_tensor, true_tensor)
        print(f"✅ SMAPE loss: {loss_value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics and losses test failed: {e}")
        return False


def test_model_components():
    """Test individual model components."""
    print("Testing individual model components...")
    
    try:
        # Test TSEmb
        ts_emb = TSEmb(lookback=96, d_model=512, num_channels=21)
        x = torch.randn(2, 96, 21)
        emb_output = ts_emb(x)
        print(f"✅ TSEmb: Input {x.shape} -> Output {emb_output.shape}")
        
        # Test CrossAttention
        cross_attn = CrossAttention(d_model=512, n_heads=8)
        queries = torch.randn(2, 21, 512)
        keys = torch.randn(2, 10, 512)
        values = torch.randn(2, 10, 512)
        attn_output = cross_attn(queries, keys, values)
        print(f"✅ CrossAttention: Output shape {attn_output.shape}")
        
        # Test PromptAlignment
        prompt_align = PromptAlignment(d_model=512, num_heads=8)
        ts_tokens = torch.randn(2, 21, 512)
        prompt_embeddings = torch.randn(2, 10, 512)
        align_output = prompt_align(ts_tokens, prompt_embeddings)
        print(f"✅ PromptAlignment: Output shape {align_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TimeLlaMA Model Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Model initialization
    model = test_timellama_model_initialization()
    test_results['initialization'] = model is not None
    
    # Test 2: Forward pass
    test_results['forward_pass'] = test_timellama_forward_pass(model)
    
    # Test 3: Ablation variants
    test_results['ablation_variants'] = test_ablation_variants()
    
    # Test 4: Embedding variants
    test_results['embedding_variants'] = test_embedding_variants()
    
    # Test 5: Metrics and losses
    test_results['metrics_losses'] = test_metrics_and_losses()
    
    # Test 6: Model components
    test_results['model_components'] = test_model_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TimeLlaMA implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
