# Hybrid Time Series Forecasting System

A sophisticated, unified approach to commodity price forecasting that combines **Time-LlaMA-style adapters**, **GraphRAG knowledge graphs**, and **PEFT/LoRA** for efficient training within GPU memory constraints.

## 🎯 Overview

This system implements a hybrid approach for the **Mitsui Commodity Prediction Challenge** that combines:

1. **Unified Configuration System**: Centralized configuration management for all components
2. **Knowledge Graph Construction**: Creates a SQLite-based knowledge graph from time series patches
3. **GraphRAG Retrieval**: Retrieves relevant context for forecasting tasks
4. **Time-LlaMA Adapter**: Patch-based transformer with cross-attention to KG context
5. **PEFT/LoRA Integration**: Efficient fine-tuning within 16GB GPU constraints
6. **Unified Pipelines**: Clean, modular training and inference pipelines

## 🏗️ Architecture

### Core Components

#### **Unified Core System** (`src/core/`)
- **Configuration Manager** (`config.py`): Centralized configuration system with validation
- **Data Processor** (`data_processor.py`): Unified data loading, preprocessing, and patch generation
- **Model Factory** (`model_factory.py`): Factory for creating and managing model components
- **Training Pipeline** (`training_pipeline.py`): End-to-end training orchestration
- **Inference Pipeline** (`inference_pipeline.py`): Complete inference and submission pipeline

#### **Specialized Components**
- **Knowledge Graph Builder** (`src/kg/graph_builder.py`): Constructs SQLite knowledge graph from time series patches
- **GraphRAG Retriever** (`src/kg/graph_rag.py`): Retrieves relevant subgraphs for forecasting
- **Patch Embedder** (`src/data/patch_embedder.py`): Converts time series windows to embeddings
- **Time-LlaMA Adapter** (`src/models/time_llama_adapter.py`): Transformer with cross-attention to KG context

### Key Features

- **Unified Configuration**: Single configuration file manages all system components
- **Patch-as-Token**: Each time series window becomes a token (7, 14, 28-day windows)
- **Cross-Attention**: Time series patches attend to knowledge graph context
- **Multi-Target Forecasting**: Predicts multiple horizons (1, 2, 3, 4-day lags)
- **SQLite Storage**: Knowledge graph stored in SQLite for Kaggle compatibility
- **PEFT/LoRA**: Efficient training with parameter-efficient fine-tuning
- **Modular Design**: Clean separation of concerns with unified interfaces

## 📊 Data Schema

### Knowledge Graph Schema

**Nodes:**
- `TS_PATCH`: Time series window embeddings
- `VAR`: Variable/instrument metadata
- `ENTITY_DATE`: Trading day metadata
- `ENTITY_MARKET`: Market/exchange information
- `ENTITY_INSTRUMENT`: Instrument-level metadata

**Edges:**
- `POSITIVE_CORR`/`NEGATIVE_CORR`: Correlation relationships
- `COINTEGRATED`: Cointegration relationships
- `TEMPORAL_NEXT`: Sequential patch linkage
- `MARKET_RELATION`: Market-instrument relationships

### SQLite Tables

```sql
-- Time series patch nodes
CREATE TABLE ts_nodes (
    node_id TEXT PRIMARY KEY,
    series_id TEXT,
    exchange TEXT,
    instrument TEXT,
    window_start INTEGER,
    window_end INTEGER,
    window_size INTEGER,
    stride INTEGER,
    embedding BLOB,
    stats_json TEXT,
    regime_label TEXT,
    created_at TEXT
);

-- Relationship edges
CREATE TABLE ts_edges (
    source_node TEXT,
    target_node TEXT,
    relation_type TEXT,
    weight REAL,
    lag INTEGER,
    p_value REAL,
    test_stat REAL,
    window_size INTEGER,
    gap_days INTEGER
);

-- Entity nodes
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT,
    attrs_json TEXT
);
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p database models configs logs outputs
```

### 2. Build Knowledge Graph

```bash
python main.py --mode build_kg --config configs/training_config.json
```

### 3. Train Model

```bash
python main.py --mode train --config configs/training_config.json
```

### 4. Run Inference

```bash
python main.py --mode inference --config configs/training_config.json
```

### 5. Run Complete Pipeline

```bash
python main.py --mode full_pipeline --config configs/training_config.json --output_dir outputs
```

## ⚙️ Configuration

### Training Configuration (`configs/training_config.json`)

```json
{
  "kg": {
    "db_path": "database/commodity_kg.db",
    "window_sizes": [7, 14, 28],
    "strides": [1, 3, 7]
  },
  "model": {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 6,
    "use_peft": true,
    "lora_r": 16,
    "lora_alpha": 32
  },
  "training": {
    "batch_size": 8,
    "max_epochs": 20,
    "learning_rate": 1e-4
  }
}
```

### Inference Configuration (`configs/inference_config.json`)

```json
{
  "model_path": "models/best_model.pth",
  "kg_db_path": "database/commodity_kg.db",
  "retrieval": {
    "max_nodes": 128,
    "max_edges": 256,
    "similarity_threshold": 0.1
  }
}
```

## 🔧 Advanced Usage

### Custom Patch Configuration

```python
from src.data.patch_embedder import PatchConfig

patch_config = PatchConfig(
    window_sizes=[7, 14, 28, 56],  # Custom window sizes
    strides=[1, 2, 4, 7],          # Custom strides
    embedding_dim=256,             # Larger embeddings
    max_patches_per_series=200     # More patches per series
)
```

### Custom Retrieval Configuration

```python
from src.kg.graph_rag import RetrievalConfig

retrieval_config = RetrievalConfig(
    max_nodes=256,                 # More context nodes
    max_edges=512,                 # More edges
    similarity_threshold=0.05,     # Lower threshold
    max_hops=3                     # Deeper graph expansion
)
```

### Custom Model Architecture

```python
from src.models.time_llama_adapter import TimeLlaMAConfig

model_config = TimeLlaMAConfig(
    d_model=256,                   # Larger model
    n_heads=8,                     # More attention heads
    n_layers=12,                   # Deeper model
    d_context=128,                 # Larger context dimension
    use_peft=True,                 # Enable PEFT
    lora_r=32,                     # Larger LoRA rank
    lora_alpha=64                  # Larger LoRA alpha
)
```

## 📈 Performance Optimization

### Memory Management

- **PEFT/LoRA**: Reduces trainable parameters by 90%+
- **Gradient Checkpointing**: Reduces memory usage during training
- **Mixed Precision**: Uses FP16 for faster training
- **Context Budgeting**: Limits KG context to fit GPU memory

### Training Efficiency

- **Batch Processing**: Efficient batching of patch sequences
- **Caching**: Caches KG retrievals for repeated queries
- **Indexed Queries**: Uses SQLite indexes for fast retrieval
- **Parallel Processing**: Multi-worker data loading

## 🧪 Evaluation

### Metrics

- **MSE**: Mean Squared Error for regression
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct direction predictions

### Validation Strategy

- **Rolling Origin**: Time series cross-validation
- **Walk-Forward**: Expanding window validation
- **Holdout**: Last 20% of data for final evaluation

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Error**: Reduce batch size or model dimensions
2. **SQLite Lock**: Ensure only one process accesses the database
3. **Empty Context**: Check KG construction and retrieval thresholds
4. **Training Instability**: Reduce learning rate or increase gradient clipping

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- **Time-LlaMA**: Patch-based time series forecasting with LLMs
- **GraphRAG**: Knowledge graph-augmented retrieval
- **PEFT**: Parameter-efficient fine-tuning
- **Lag-Llama**: Probabilistic time series forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Mitsui & Co. for the commodity prediction challenge
- The Lag-Llama team for the base architecture
- The PEFT team for efficient fine-tuning tools
- The GraphRAG community for knowledge graph techniques
