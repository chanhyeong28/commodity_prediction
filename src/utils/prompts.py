"""
Prompt utilities for TimeLlaMA and Time-LLM
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


def build_prompt_embeddings(prompts: List[str], tokenizer, llm_model, device: torch.device) -> torch.Tensor:
    """
    Build prompt embeddings from text prompts using frozen LLM components.
    
    Args:
        prompts: List of text prompts (one per batch sample)
        tokenizer: LLM tokenizer
        llm_model: LLM model
        device: Device to place tensors on
        
    Returns:
        Prompt embeddings tensor [B, P, d_llm]
    """
    # Tokenize prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = tokenized.input_ids.to(device)
    
    # Get embeddings from frozen LLM
    with torch.no_grad():
        prompt_embeddings = llm_model.get_input_embeddings()(input_ids)
    
    return prompt_embeddings


def create_forecasting_prompt(
    seq_len: int, 
    pred_len: int, 
    min_val: float, 
    max_val: float, 
    median_val: float, 
    trend: str, 
    lags: List[int],
    description: str = "Time series forecasting task"
) -> str:
    """
    Create a forecasting prompt with data statistics.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        min_val: Minimum value in the sequence
        max_val: Maximum value in the sequence
        median_val: Median value in the sequence
        trend: Trend direction ('upward' or 'downward')
        lags: Top lag values
        description: Dataset description
        
    Returns:
        Formatted prompt string
    """
    lags_str = str(lags)
    
    prompt = (
        f"<|start_prompt|>Dataset description: {description}"
        f"Task description: forecast the next {pred_len} steps given the previous {seq_len} steps information; "
        "Input statistics: "
        f"min value {min_val}, "
        f"max value {max_val}, "
        f"median value {median_val}, "
        f"the trend of input is {trend}, "
        f"top 5 lags are : {lags_str}<|<end_prompt>|>"
    )
    
    return prompt


def extract_time_series_stats(x: torch.Tensor) -> Dict[str, Any]:
    """
    Extract statistics from time series data.
    
    Args:
        x: Time series tensor [B, T, N] or [B*N, T, 1]
        
    Returns:
        Dictionary containing statistics
    """
    min_values = torch.min(x, dim=1)[0]
    max_values = torch.max(x, dim=1)[0]
    medians = torch.median(x, dim=1).values
    trends = x.diff(dim=1).sum(dim=1)
    
    return {
        'min_values': min_values,
        'max_values': max_values,
        'medians': medians,
        'trends': trends
    }


def create_domain_specific_prompts(
    domain: str,
    seq_len: int,
    pred_len: int,
    stats: Dict[str, Any]
) -> List[str]:
    """
    Create domain-specific prompts for different datasets.
    
    Args:
        domain: Dataset domain ('mitsui', 'ett', 'ecl', etc.)
        seq_len: Input sequence length
        pred_len: Prediction length
        stats: Time series statistics
        
    Returns:
        List of formatted prompts
    """
    prompts = []
    
    for i in range(len(stats['min_values'])):
        min_val = stats['min_values'][i].item()
        max_val = stats['max_values'][i].item()
        median_val = stats['medians'][i].item()
        trend = 'upward' if stats['trends'][i] > 0 else 'downward'
        
        # Domain-specific descriptions
        descriptions = {
            'mitsui': 'The Mitsui commodity prediction dataset is a collection of financial instruments and commodity price data.',
            'ett': 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.',
            'ecl': 'The ECL dataset contains electricity consumption data.',
            'traffic': 'The Traffic dataset contains traffic flow data.',
            'weather': 'The Weather dataset contains weather measurement data.'
        }
        
        description = descriptions.get(domain, 'Time series forecasting task')
        
        # Create lags (simplified for now)
        lags = [1, 2, 3, 4, 5]
        
        prompt = create_forecasting_prompt(
            seq_len=seq_len,
            pred_len=pred_len,
            min_val=min_val,
            max_val=max_val,
            median_val=median_val,
            trend=trend,
            lags=lags,
            description=description
        )
        
        prompts.append(prompt)
    
    return prompts
