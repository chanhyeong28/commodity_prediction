#!/usr/bin/env python3
"""
Quick Test Script - Focus on Core Functionality

This script quickly validates that all components can be imported and basic functionality works
without getting stuck in heavy data processing.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from src.core.config import SystemConfig, ConfigManager
        print("✅ Core config imported")
        
        from src.core.data_processor import CommodityDataProcessor
        print("✅ Data processor imported")
        
        from src.core.model_factory import ModelFactory
        print("✅ Model factory imported")
        
        from src.core.training_pipeline import create_training_pipeline
        print("✅ Training pipeline imported")
        
        from src.core.inference_pipeline import create_inference_pipeline
        print("✅ Inference pipeline imported")
        
        from src.models.time_llama_adapter import TimeLlaMAWithPEFT, TimeLlaMAConfig
        print("✅ Time-LlaMA model imported")
        
        from src.kg.graph_builder import KnowledgeGraphBuilder
        print("✅ Knowledge graph builder imported")
        
        from src.kg.graph_rag import GraphRAGRetriever
        print("✅ GraphRAG retriever imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_system():
    """Test configuration system"""
    print("\n⚙️  Testing configuration system...")
    
    try:
        from src.core.config import SystemConfig, ConfigManager
        
        # Test default config
        config = SystemConfig()
        assert config.model.d_model == 128
        print("✅ Default config works")
        
        # Test config manager
        manager = ConfigManager()
        manager.config = config
        assert manager.validate_config()
        print("✅ Config validation works")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_creation():
    """Test model creation without heavy processing"""
    print("\n🤖 Testing model creation...")
    
    try:
        from src.core.config import SystemConfig
        from src.core.model_factory import ModelFactory
        
        # Create minimal config
        config = SystemConfig()
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.use_peft = False
        
        # Create model
        factory = ModelFactory(config)
        model = factory.create_time_llama_model()
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 10
        context_len = 5
        device = factory.device
        
        patch_tokens = torch.randn(batch_size, seq_len, config.model.d_model).to(device)
        context_tokens = torch.randn(batch_size, context_len, config.model.d_context).to(device)
        context_mask = torch.ones(batch_size, context_len).to(device)
        # Create proper attention mask for self-attention (square matrix)
        attention_mask = torch.ones(seq_len, seq_len).to(device)
        
        with torch.no_grad():
            forecasts = model(patch_tokens, context_tokens, context_mask, attention_mask)
        
        print(f"✅ Model created and forward pass works")
        print(f"   - Total parameters: {model.get_num_parameters():,}")
        print(f"   - Trainable parameters: {model.get_trainable_parameters():,}")
        print(f"   - Output horizons: {list(forecasts.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_processing():
    """Test data processing with minimal data"""
    print("\n📊 Testing data processing...")
    
    try:
        from src.core.config import SystemConfig
        from src.core.data_processor import CommodityDataProcessor
        
        # Create minimal config
        config = SystemConfig()
        config.patches.window_sizes = [7]
        config.patches.max_patches_per_series = 5
        
        # Create minimal test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'date_id': dates,
            'LME_AL_Close': np.random.randn(50).cumsum() + 100,
            'JPX_Gold_Close': np.random.randn(50).cumsum() + 2000
        })
        
        # Test data processor
        processor = CommodityDataProcessor(config)
        processed_data = processor.preprocess_data(test_data)
        
        # Test patch creation
        patches = processor.create_patches(processed_data, 'LME_AL_Close')
        
        print(f"✅ Data processing works")
        print(f"   - Processed data shape: {processed_data.shape}")
        print(f"   - Created {len(patches)} patches")
        
        return True
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation without running them"""
    print("\n🔄 Testing pipeline creation...")
    
    try:
        from src.core.config import SystemConfig
        from src.core.training_pipeline import create_training_pipeline
        from src.core.inference_pipeline import create_inference_pipeline
        
        # Create minimal config
        config = SystemConfig()
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.use_peft = False
        config.training.max_epochs = 1
        config.training.batch_size = 1
        
        # Test pipeline creation
        train_pipeline = create_training_pipeline(config)
        inference_pipeline = create_inference_pipeline(config)
        
        print("✅ Pipeline creation works")
        print("   - Training pipeline created")
        print("   - Inference pipeline created")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all quick tests"""
    print("🚀 Quick System Validation Test\n")
    
    tests = [
        test_imports,
        test_config_system,
        test_model_creation,
        test_data_processing,
        test_pipeline_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Quick Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quick tests passed! System is ready for development.")
        print("\n✅ System Status:")
        print("   - All imports work correctly")
        print("   - Configuration system is functional")
        print("   - Model creation and forward pass work")
        print("   - Data processing pipeline is functional")
        print("   - Training and inference pipelines can be created")
        print("\n🚀 Ready for next phase: Training and optimization!")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()
