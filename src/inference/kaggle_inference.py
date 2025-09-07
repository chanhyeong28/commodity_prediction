"""
Kaggle Inference Pipeline for Commodity Forecasting

This module implements the inference pipeline specifically designed for Kaggle
submission, handling the evaluation API and producing forecasts for the test set.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import sqlite3
import pickle
from tqdm import tqdm

# Import our modules
from ..kg.graph_rag import GraphRAGRetriever, RetrievalConfig
from ..data.patch_embedder import PatchEmbeddingPipeline, PatchConfig
from ..models.time_llama_adapter import TimeLlaMAWithPEFT, TimeLlaMAConfig

logger = logging.getLogger(__name__)

class KaggleInferenceEngine:
    """
    Inference engine for Kaggle commodity forecasting competition.
    
    This engine handles:
    1. Loading the pre-trained model and knowledge graph
    2. Processing test data from the evaluation API
    3. Retrieving relevant context for each forecast
    4. Generating predictions for all target pairs
    5. Formatting output for submission
    """
    
    def __init__(self, 
                 model_path: str,
                 kg_db_path: str,
                 config: Dict[str, Any]):
        self.model_path = model_path
        self.kg_db_path = kg_db_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._load_model()
        self._setup_retrieval()
        self._setup_patch_pipeline()
        
        # Load target pairs
        self.target_pairs = self._load_target_pairs()
        
    def _load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with same config
        model_config = TimeLlaMAConfig(**self.config['model'])
        self.model = TimeLlaMAWithPEFT(model_config)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _setup_retrieval(self):
        """Setup GraphRAG retrieval system"""
        retrieval_config = RetrievalConfig(**self.config['retrieval'])
        
        self.kg_retriever = GraphRAGRetriever(
            db_path=self.kg_db_path,
            config=retrieval_config
        )
        
        logger.info("GraphRAG retriever setup complete")
    
    def _setup_patch_pipeline(self):
        """Setup patch embedding pipeline"""
        patch_config = PatchConfig(**self.config['patches'])
        
        self.patch_pipeline = PatchEmbeddingPipeline(
            patch_config=patch_config,
            embedder_config=self.config['patches']['embedder_config']
        )
        
        logger.info("Patch embedding pipeline setup complete")
    
    def _load_target_pairs(self) -> pd.DataFrame:
        """Load target pairs configuration"""
        target_pairs_path = '/data/kaggle_projects/commodity_prediction/kaggle_data/target_pairs.csv'
        target_pairs = pd.read_csv(target_pairs_path)
        
        logger.info(f"Loaded {len(target_pairs)} target pairs")
        return target_pairs
    
    def _parse_target(self, target: str) -> str:
        """Parse target string to get primary series"""
        if ' - ' in target:
            # Difference target: "LME_AL_Close - US_Stock_VT_adj_close"
            parts = target.split(' - ')
            return parts[0].strip()
        else:
            return target
    
    def _get_latest_data(self, all_data: pd.DataFrame, current_date: int) -> pd.DataFrame:
        """Get the latest available data up to current date"""
        return all_data[all_data['date_id'] <= current_date]
    
    def _prepare_patch_embeddings(self, data: pd.DataFrame, target_series: str) -> torch.Tensor:
        """Prepare patch embeddings for a target series"""
        try:
            # Transform data to get patch embeddings
            embeddings = self.patch_pipeline.transform(
                data,
                target_series=[target_series]
            )
            
            if target_series in embeddings:
                return embeddings[target_series]['embeddings']
            else:
                # Fallback: create dummy embeddings
                dummy_embeddings = torch.zeros(
                    1, 10, self.patch_pipeline.get_embedding_dim()
                ).to(self.device)
                return dummy_embeddings
                
        except Exception as e:
            logger.warning(f"Error creating patch embeddings for {target_series}: {e}")
            # Fallback: create dummy embeddings
            dummy_embeddings = torch.zeros(
                1, 10, self.patch_pipeline.get_embedding_dim()
            ).to(self.device)
            return dummy_embeddings
    
    def _prepare_kg_context(self, target_series: str, data: pd.DataFrame, 
                          forecast_date: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare knowledge graph context for a target series"""
        try:
            # Get recent data for the target series
            series_data = data[target_series].dropna().tail(30)
            
            if len(series_data) < 7:  # Minimum data requirement
                # Return empty context
                context_tokens = torch.zeros(1, 1, self.config['model']['d_context']).to(self.device)
                context_mask = torch.zeros(1, 1).to(self.device)
                return context_tokens, context_mask
            
            # Retrieve KG context
            kg_result = self.kg_retriever.retrieve(
                target_series,
                series_data,
                forecast_date
            )
            
            # Prepare context for model
            context_data = self.kg_retriever.prepare_context_for_model(kg_result)
            
            if context_data['nodes']:
                # Convert context to tensors
                context_tokens = torch.tensor([
                    node['embedding'] for node in context_data['nodes']
                ]).unsqueeze(0).to(self.device)
                context_mask = torch.ones(1, len(context_data['nodes'])).to(self.device)
            else:
                # Empty context
                context_tokens = torch.zeros(1, 1, self.config['model']['d_context']).to(self.device)
                context_mask = torch.zeros(1, 1).to(self.device)
            
            return context_tokens, context_mask
            
        except Exception as e:
            logger.warning(f"Error retrieving KG context for {target_series}: {e}")
            # Return empty context
            context_tokens = torch.zeros(1, 1, self.config['model']['d_context']).to(self.device)
            context_mask = torch.zeros(1, 1).to(self.device)
            return context_tokens, context_mask
    
    def predict_single_target(self, 
                            target: str,
                            lag: int,
                            all_data: pd.DataFrame,
                            current_date: int) -> float:
        """
        Predict a single target for a specific lag.
        
        Args:
            target: Target string (e.g., "LME_AL_Close - US_Stock_VT_adj_close")
            lag: Forecast lag
            all_data: All available data
            current_date: Current date for forecasting
            
        Returns:
            Predicted value
        """
        # Get latest data
        latest_data = self._get_latest_data(all_data, current_date)
        
        # Parse target to get primary series
        target_series = self._parse_target(target)
        
        # Prepare patch embeddings
        patch_embeddings = self._prepare_patch_embeddings(latest_data, target_series)
        
        # Prepare KG context
        context_tokens, context_mask = self._prepare_kg_context(
            target_series, latest_data, current_date
        )
        
        # Create attention mask for patches
        attention_mask = torch.ones(1, patch_embeddings.size(1)).to(self.device)
        
        # Generate forecast
        with torch.no_grad():
            forecasts = self.model(
                patch_embeddings,
                context_tokens,
                context_mask,
                attention_mask
            )
        
        # Get forecast for the specific lag
        horizon_key = f'horizon_{lag}'
        if horizon_key in forecasts:
            # For multi-target forecasting, we need to map to the specific target
            # This is a simplified approach - in practice, you'd need proper target mapping
            forecast = forecasts[horizon_key].squeeze().item()
        else:
            # Fallback: use the first available horizon
            available_horizons = list(forecasts.keys())
            if available_horizons:
                forecast = forecasts[available_horizons[0]].squeeze().item()
            else:
                forecast = 0.0  # Default fallback
        
        return forecast
    
    def predict_batch(self, 
                     test_data: pd.DataFrame,
                     all_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a batch of test data.
        
        Args:
            test_data: Test data from evaluation API
            all_data: All available data (train + test)
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        
        # Group by date for efficient processing
        for date_id in tqdm(test_data['date_id'].unique(), desc="Processing dates"):
            date_data = test_data[test_data['date_id'] == date_id]
            
            for _, row in date_data.iterrows():
                # Get target and lag from target_pairs
                target_row = self.target_pairs[
                    self.target_pairs['target'] == row['target']
                ].iloc[0]
                
                target = target_row['target']
                lag = target_row['lag']
                
                # Generate prediction
                try:
                    prediction = self.predict_single_target(
                        target, lag, all_data, date_id
                    )
                except Exception as e:
                    logger.error(f"Error predicting {target} at {date_id}: {e}")
                    prediction = 0.0  # Fallback
                
                predictions.append({
                    'date_id': date_id,
                    'target': target,
                    'prediction': prediction
                })
        
        return pd.DataFrame(predictions)
    
    def run_inference(self, 
                     test_data: pd.DataFrame,
                     all_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main inference method for Kaggle submission.
        
        Args:
            test_data: Test data from evaluation API
            all_data: All available data
            
        Returns:
            DataFrame with predictions in submission format
        """
        logger.info("Starting inference...")
        
        # Generate predictions
        predictions_df = self.predict_batch(test_data, all_data)
        
        # Format for submission
        submission_df = predictions_df.pivot(
            index='date_id',
            columns='target',
            values='prediction'
        ).reset_index()
        
        # Ensure all targets are present
        for target in self.target_pairs['target']:
            if target not in submission_df.columns:
                submission_df[target] = 0.0
        
        # Reorder columns to match expected format
        target_columns = ['date_id'] + list(self.target_pairs['target'])
        submission_df = submission_df[target_columns]
        
        logger.info(f"Generated predictions for {len(submission_df)} dates")
        
        return submission_df

class KaggleSubmissionHandler:
    """
    Handler for Kaggle submission process.
    
    This class manages the interaction with Kaggle's evaluation API
    and coordinates the inference process.
    """
    
    def __init__(self, inference_engine: KaggleInferenceEngine):
        self.inference_engine = inference_engine
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        # Load training data
        train_data = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/train.csv')
        
        # Load test data (this would come from Kaggle API in practice)
        test_data = pd.read_csv('/data/kaggle_projects/commodity_prediction/kaggle_data/test.csv')
        
        return train_data, test_data
    
    def run_submission(self) -> pd.DataFrame:
        """Run the complete submission process"""
        logger.info("Starting Kaggle submission process...")
        
        # Load data
        train_data, test_data = self.load_data()
        
        # Combine all data for context
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Run inference
        predictions = self.inference_engine.run_inference(test_data, all_data)
        
        # Save predictions
        output_path = '/data/kaggle_projects/commodity_prediction/submission.csv'
        predictions.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        
        return predictions

def create_inference_engine(config_path: str = None) -> KaggleInferenceEngine:
    """
    Factory function to create inference engine.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured inference engine
    """
    if config_path is None:
        config_path = '/data/kaggle_projects/commodity_prediction/configs/inference_config.json'
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create inference engine
    engine = KaggleInferenceEngine(
        model_path=config['model_path'],
        kg_db_path=config['kg_db_path'],
        config=config
    )
    
    return engine

def main():
    """Example inference script"""
    
    # Configuration
    config = {
        'model_path': '/data/kaggle_projects/commodity_prediction/models/best_model.pth',
        'kg_db_path': '/data/kaggle_projects/commodity_prediction/database/commodity_kg.db',
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 6,
            'd_ff': 512,
            'd_context': 64,
            'n_context_heads': 2,
            'n_targets': 80,
            'forecast_horizons': [1, 2, 3, 4],
            'use_peft': True,
            'lora_r': 16,
            'lora_alpha': 32
        },
        'retrieval': {
            'max_nodes': 128,
            'max_edges': 256,
            'similarity_threshold': 0.1
        },
        'patches': {
            'window_sizes': [7, 14, 28],
            'strides': [1, 3, 7],
            'embedding_dim': 128,
            'embedder_config': {
                'architecture': 'hybrid',
                'dropout': 0.1
            }
        }
    }
    
    # Create inference engine
    engine = KaggleInferenceEngine(
        model_path=config['model_path'],
        kg_db_path=config['kg_db_path'],
        config=config
    )
    
    # Create submission handler
    handler = KaggleSubmissionHandler(engine)
    
    # Run submission
    predictions = handler.run_submission()
    
    print(f"Generated predictions for {len(predictions)} rows")
    print(predictions.head())

if __name__ == "__main__":
    main()
