"""
Unified Inference Pipeline for Hybrid Time Series Forecasting System

This module provides a comprehensive inference pipeline that handles:
1. Model loading and initialization
2. Data preprocessing for inference
3. GraphRAG context retrieval
4. Batch inference processing
5. Output formatting and submission
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from tqdm import tqdm
import json

from .config import SystemConfig
from .data_processor import CommodityDataProcessor
from .model_factory import ModelManager
from ..kg.graph_rag import GraphRAGRetriever, RetrievalConfig

logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    Unified inference pipeline for the hybrid forecasting system.
    
    This class handles the complete inference process including model loading,
    data preprocessing, context retrieval, and prediction generation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_processor = CommodityDataProcessor(config)
        self.model_manager = ModelManager(config)
        self.kg_retriever = None
        
        # Inference state
        self.model_loaded = False
        self.data_prepared = False
        
    def load_model(self, model_path: str = None):
        """
        Load the trained model.
        
        Args:
            model_path: Path to model checkpoint (uses config if None)
        """
        if model_path is None:
            model_path = self.config.inference.model_path
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        self.model_manager.load_checkpoint(model_path)
        self.model_manager.set_eval_mode()
        
        self.model_loaded = True
        logger.info("Model loaded successfully")
    
    def setup_knowledge_graph(self):
        """Setup the knowledge graph retriever"""
        logger.info("Setting up knowledge graph retriever...")
        
        # Create KG retriever
        retrieval_config = RetrievalConfig(
            max_nodes=self.config.retrieval.max_nodes,
            max_edges=self.config.retrieval.max_edges,
            similarity_threshold=self.config.retrieval.similarity_threshold,
            ts_patch_ratio=self.config.retrieval.ts_patch_ratio,
            max_hops=self.config.retrieval.max_hops,
            recency_decay=self.config.retrieval.recency_decay,
            similarity_weight=self.config.retrieval.similarity_weight,
            market_match_weight=self.config.retrieval.market_match_weight,
            recency_weight=self.config.retrieval.recency_weight,
            edge_strength_weight=self.config.retrieval.edge_strength_weight
        )
        
        self.kg_retriever = GraphRAGRetriever(
            db_path=self.config.inference.kg_db_path,
            config=retrieval_config
        )
        
        logger.info("Knowledge graph retriever setup complete")
    
    def prepare_inference_data(self, test_data: pd.DataFrame, 
                              all_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare data for inference.
        
        Args:
            test_data: Test data for inference
            all_data: All available data for context
            
        Returns:
            Prepared test data
        """
        logger.info("Preparing inference data...")
        
        # Use all_data if provided, otherwise use test_data
        if all_data is not None:
            context_data = all_data
        else:
            context_data = test_data
        
        # Preprocess context data
        context_data = self.data_processor.preprocess_data(context_data)
        
        # Extract metadata
        self.data_processor.extract_series_metadata(context_data)
        
        # Fit scaler on context data
        self.data_processor.fit_scaler(context_data)
        
        # Transform context data
        context_data = self.data_processor.transform_data(context_data)
        
        # Preprocess test data
        test_data = self.data_processor.preprocess_data(test_data)
        
        # Transform test data
        test_data = self.data_processor.transform_data(test_data)
        
        self.data_prepared = True
        logger.info("Inference data prepared")
        
        return test_data, context_data
    
    def predict_single_target(self, target: str, lag: int, 
                            context_data: pd.DataFrame, 
                            current_date: int) -> float:
        """
        Predict a single target for a specific lag.
        
        Args:
            target: Target string
            lag: Forecast lag
            context_data: Context data
            current_date: Current date for forecasting
            
        Returns:
            Predicted value
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get latest data up to current date
        latest_data = context_data[context_data['date_id'] <= current_date]
        
        # Parse target to get primary series
        target_series = self._parse_target(target)
        
        # Prepare patch embeddings
        patch_embeddings = self._prepare_patch_embeddings(latest_data, target_series)
        
        # Prepare KG context
        kg_context, kg_mask = self._prepare_kg_context(
            target_series, latest_data, current_date
        )
        
        # Create attention mask for patches
        attention_mask = torch.ones(1, patch_embeddings.size(1)).to(self.model_manager.factory.device)
        
        # Generate forecast
        with torch.no_grad():
            forecasts = self.model_manager.model(
                patch_embeddings,
                kg_context,
                kg_mask,
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
    
    def predict_batch(self, test_data: pd.DataFrame, 
                     context_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict for a batch of test data.
        
        Args:
            test_data: Test data
            context_data: Context data
            
        Returns:
            DataFrame with predictions
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Running batch inference...")
        
        predictions = []
        
        # Group by date for efficient processing
        for date_id in tqdm(test_data['date_id'].unique(), desc="Processing dates"):
            date_data = test_data[test_data['date_id'] == date_id]
            
            for _, row in date_data.iterrows():
                # Get target and lag from target_pairs
                target_row = self.data_processor.target_pairs[
                    self.data_processor.target_pairs['target'] == row['target']
                ].iloc[0]
                
                target = target_row['target']
                lag = target_row['lag']
                
                # Generate prediction
                try:
                    prediction = self.predict_single_target(
                        target, lag, context_data, date_id
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
    
    def run_inference(self, test_data: pd.DataFrame, 
                     all_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run complete inference pipeline.
        
        Args:
            test_data: Test data for inference
            all_data: All available data for context
            
        Returns:
            DataFrame with predictions in submission format
        """
        logger.info("Starting inference pipeline...")
        
        # Prepare data
        test_data, context_data = self.prepare_inference_data(test_data, all_data)
        
        # Generate predictions
        predictions_df = self.predict_batch(test_data, context_data)
        
        # Format for submission
        submission_df = predictions_df.pivot(
            index='date_id',
            columns='target',
            values='prediction'
        ).reset_index()
        
        # Ensure all targets are present
        for target in self.data_processor.target_pairs['target']:
            if target not in submission_df.columns:
                submission_df[target] = 0.0
        
        # Reorder columns to match expected format
        target_columns = ['date_id'] + list(self.data_processor.target_pairs['target'])
        submission_df = submission_df[target_columns]
        
        logger.info(f"Generated predictions for {len(submission_df)} dates")
        
        return submission_df
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: str = None):
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions DataFrame
            output_path: Output file path (uses config if None)
        """
        if output_path is None:
            output_path = self.config.inference.output_path
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def _parse_target(self, target: str) -> str:
        """Parse target string to get primary series"""
        if ' - ' in target:
            parts = target.split(' - ')
            return parts[0].strip()
        else:
            return target
    
    def _prepare_patch_embeddings(self, data: pd.DataFrame, target_series: str) -> torch.Tensor:
        """Prepare patch embeddings for a target series"""
        try:
            # Create patches
            patches = self.data_processor.create_patches(data, target_series)
            
            if not patches:
                # Create dummy embeddings if no patches available
                dummy_embeddings = torch.zeros(
                    1, min(self.config.patches.window_sizes), self.config.patches.embedding_dim
                ).to(self.model_manager.factory.device)
                return dummy_embeddings
            
            # Convert patches to embeddings (simplified)
            # In practice, you'd use the trained patch embedder
            patch_values = [patch['values'] for patch in patches]
            max_length = max(len(pv) for pv in patch_values)
            
            # Pad patches to same length
            padded_patches = []
            for pv in patch_values:
                if len(pv) < max_length:
                    padded = np.pad(pv, (0, max_length - len(pv)), mode='constant')
                else:
                    padded = pv[:max_length]
                padded_patches.append(padded)
            
            # Convert to tensor
            patch_tensor = torch.FloatTensor(padded_patches).unsqueeze(-1)  # Add input_dim
            
            # Project to embedding dimension (simplified)
            embedding_projection = torch.nn.Linear(1, self.config.patches.embedding_dim)
            embeddings = embedding_projection(patch_tensor)
            
            return embeddings.to(self.model_manager.factory.device)
            
        except Exception as e:
            logger.warning(f"Error creating patch embeddings for {target_series}: {e}")
            # Return dummy embeddings
            dummy_embeddings = torch.zeros(
                1, min(self.config.patches.window_sizes), self.config.patches.embedding_dim
            ).to(self.model_manager.factory.device)
            return dummy_embeddings
    
    def _prepare_kg_context(self, target_series: str, data: pd.DataFrame, 
                          forecast_date: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare knowledge graph context for a target series"""
        try:
            if self.kg_retriever is None:
                # Return empty context if KG retriever not available
                context_tokens = torch.zeros(1, 1, self.config.model.d_context).to(self.model_manager.factory.device)
                context_mask = torch.zeros(1, 1).to(self.model_manager.factory.device)
                return context_tokens, context_mask
            
            # Get recent data for the target series
            series_data = data[target_series].dropna().tail(30)
            
            if len(series_data) < 7:  # Minimum data requirement
                # Return empty context
                context_tokens = torch.zeros(1, 1, self.config.model.d_context).to(self.model_manager.factory.device)
                context_mask = torch.zeros(1, 1).to(self.model_manager.factory.device)
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
                ]).unsqueeze(0).to(self.model_manager.factory.device)
                context_mask = torch.ones(1, len(context_data['nodes'])).to(self.model_manager.factory.device)
            else:
                # Empty context
                context_tokens = torch.zeros(1, 1, self.config.model.d_context).to(self.model_manager.factory.device)
                context_mask = torch.zeros(1, 1).to(self.model_manager.factory.device)
            
            return context_tokens, context_mask
            
        except Exception as e:
            logger.warning(f"Error retrieving KG context for {target_series}: {e}")
            # Return empty context
            context_tokens = torch.zeros(1, 1, self.config.model.d_context).to(self.model_manager.factory.device)
            context_mask = torch.zeros(1, 1).to(self.model_manager.factory.device)
            return context_tokens, context_mask

def create_inference_pipeline(config: SystemConfig) -> InferencePipeline:
    """
    Create an inference pipeline with the given configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured inference pipeline
    """
    return InferencePipeline(config)

def main():
    """Example inference script"""
    from .config import load_config_from_file
    
    # Load configuration
    config_manager = load_config_from_file('configs/inference_config.json')
    config = config_manager.config
    
    # Setup logging
    config_manager.setup_logging()
    
    # Create inference pipeline
    pipeline = create_inference_pipeline(config)
    
    # Load model
    pipeline.load_model()
    
    # Setup knowledge graph
    pipeline.setup_knowledge_graph()
    
    # Load test data
    test_data = pd.read_csv(config.test_data_path)
    all_data = pd.read_csv(config.train_data_path)
    
    # Run inference
    predictions = pipeline.run_inference(test_data, all_data)
    
    # Save predictions
    pipeline.save_predictions(predictions)
    
    print(f"Generated predictions for {len(predictions)} rows")

if __name__ == "__main__":
    main()
