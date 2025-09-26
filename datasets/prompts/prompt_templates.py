#!/usr/bin/env python3
"""
Prompt template generator for TimeLlaMA.

This module provides functions to generate prompts in the TimeLlaMA format
for different scenarios and datasets.
"""

import random
from typing import Dict, List, Any, Optional
import numpy as np


class TimeLlaMAPromptGenerator:
    """Generator for TimeLlaMA-style prompts."""
    
    def __init__(self):
        self.dataset_descriptions = [
            "Commodity price forecasting task",
            "Financial time series forecasting",
            "Energy demand prediction",
            "Agricultural commodity prediction",
            "Metal price forecasting",
            "Oil and gas price prediction",
            "Currency exchange rate forecasting",
            "Stock market prediction",
            "Cryptocurrency price forecasting",
            "Real estate price prediction",
            "Consumer demand forecasting",
            "Production output prediction",
            "Sales forecasting",
            "Revenue prediction",
            "Cost forecasting",
            "Mitsui commodity price prediction challenge"
        ]
        
        self.task_descriptions = [
            "forecast the next {pred_len} steps given the previous {seq_len} steps information",
            "predict next {pred_len} time periods of commodity prices",
            "analyze market trends and predict price movements",
            "forecast demand for agricultural products",
            "predict energy consumption patterns",
            "forecast financial market indicators",
            "predict supply chain disruptions",
            "analyze seasonal patterns in commodity prices",
            "forecast inflation rates based on commodity prices",
            "predict currency fluctuations",
            "forecast economic growth indicators",
            "predict market volatility"
        ]
        
        self.trend_directions = [
            "upward", "downward", "sideways", "volatile", "stable",
            "increasing", "decreasing", "fluctuating", "trending up",
            "trending down", "consolidating", "breaking out", "breaking down",
            "reversing", "continuing", "accelerating", "decelerating",
            "oscillating", "mean reverting", "momentum building"
        ]
    
    def generate_statistics(self, 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          median_val: Optional[float] = None,
                          trend: Optional[str] = None,
                          lags: Optional[List[int]] = None) -> Dict[str, Any]:
        """Generate realistic statistics for prompts."""
        
        if min_val is None:
            min_val = round(random.uniform(-10.0, 0.0), 2)
        if max_val is None:
            max_val = round(random.uniform(0.0, 15.0), 2)
        if median_val is None:
            median_val = round(random.uniform(min_val, max_val), 2)
        if trend is None:
            trend = random.choice(self.trend_directions)
        if lags is None:
            lags = sorted(random.sample(range(1, 121), 5))
        
        return {
            "min_value": min_val,
            "max_value": max_val,
            "median_value": median_val,
            "trend": trend,
            "top_5_lags": lags,
            "variance": round(random.uniform(0.5, 5.0), 2),
            "std_dev": round(random.uniform(0.7, 2.2), 2),
            "mean": round(random.uniform(min_val, max_val), 2),
            "skewness": round(random.uniform(-1.0, 1.0), 2),
            "kurtosis": round(random.uniform(2.0, 5.0), 2),
            "autocorrelation": round(random.uniform(0.1, 0.9), 2),
            "seasonality": random.choice([True, False]),
            "trend_strength": random.choice(["weak", "moderate", "strong"]),
            "volatility": random.choice(["low", "medium", "high", "very_high"]),
            "momentum": random.choice(["positive", "negative", "neutral", "mixed"])
        }
    
    def generate_prompt(self,
                       dataset_description: str,
                       task_description: str,
                       statistics: Dict[str, Any],
                       pred_len: int = 96,
                       seq_len: int = 96) -> str:
        """Generate a complete TimeLlaMA prompt."""
        
        # Format task description with parameters
        formatted_task = task_description.format(pred_len=pred_len, seq_len=seq_len)
        
        # Build the prompt
        prompt = (
            f"<|start_prompt|>Dataset description: {dataset_description} "
            f"Task description: {formatted_task}; "
            f"Input statistics: "
            f"min value {statistics['min_value']}, "
            f"max value {statistics['max_value']}, "
            f"median value {statistics['median_value']}, "
            f"the trend of input is {statistics['trend']}, "
            f"top 5 lags are : {statistics['top_5_lags']}<|<end_prompt|>"
        )
        
        return prompt
    
    def generate_random_prompt(self, 
                             pred_len: int = 96, 
                             seq_len: int = 96) -> Dict[str, Any]:
        """Generate a random prompt with statistics."""
        
        dataset_desc = random.choice(self.dataset_descriptions)
        task_desc = random.choice(self.task_descriptions)
        statistics = self.generate_statistics()
        
        prompt = self.generate_prompt(dataset_desc, task_desc, statistics, pred_len, seq_len)
        
        return {
            "dataset_description": dataset_desc,
            "task_description": task_desc,
            "statistics": statistics,
            "prompt": prompt,
            "pred_len": pred_len,
            "seq_len": seq_len
        }
    
    def generate_mitsui_prompts(self, num_prompts: int = 10) -> List[Dict[str, Any]]:
        """Generate prompts specifically for Mitsui dataset."""
        
        prompts = []
        pred_lengths = [96, 192, 336, 720]
        
        for i in range(num_prompts):
            pred_len = random.choice(pred_lengths)
            seq_len = 96  # Fixed for Mitsui
            
            # Mitsui-specific statistics
            statistics = self.generate_statistics(
                min_val=round(random.uniform(-5.0, 0.0), 2),
                max_val=round(random.uniform(0.0, 10.0), 2),
                median_val=round(random.uniform(-2.0, 3.0), 2),
                trend=random.choice(["upward", "downward", "sideways", "volatile"]),
                lags=sorted(random.sample(range(1, 97), 5))
            )
            
            prompt_data = self.generate_prompt(
                "Mitsui commodity price prediction challenge",
                "forecast the next {pred_len} steps given the previous {seq_len} steps information",
                statistics,
                pred_len,
                seq_len
            )
            
            prompts.append({
                "id": f"mitsui_prompt_{i+1:03d}",
                "dataset_description": "Mitsui commodity price prediction challenge",
                "task_description": f"forecast the next {pred_len} steps given the previous {seq_len} steps information",
                "statistics": statistics,
                "prompt": prompt_data,
                "pred_len": pred_len,
                "seq_len": seq_len
            })
        
        return prompts
    
    def generate_ablation_prompts(self) -> Dict[str, str]:
        """Generate prompts for ablation study variants."""
        
        base_statistics = self.generate_statistics(
            min_val=-2.45,
            max_val=3.21,
            median_val=0.12,
            trend="upward",
            lags=[1, 7, 14, 21, 30]
        )
        
        base_prompt = self.generate_prompt(
            "Commodity price forecasting task",
            "forecast the next {pred_len} steps given the previous {seq_len} steps information",
            base_statistics,
            96,
            96
        )
        
        return {
            "full": base_prompt,
            "variant_1": "# No modality alignment - direct feed to LLM",
            "variant_2": f"{base_prompt} [TIME_SERIES_DATA]",
            "variant_3": base_prompt,
            "variant_4": base_prompt,
            "variant_5": base_prompt,
            "variant_6": base_prompt
        }
    
    def generate_scenario_prompts(self) -> Dict[str, str]:
        """Generate prompts for different market scenarios."""
        
        scenarios = {
            "high_volatility": {
                "min_val": -8.92,
                "max_val": 12.45,
                "median_val": 0.67,
                "trend": "volatile",
                "lags": [1, 2, 3, 4, 5]
            },
            "low_volatility": {
                "min_val": 0.12,
                "max_val": 0.89,
                "median_val": 0.45,
                "trend": "stable",
                "lags": [24, 48, 72, 96, 120]
            },
            "strong_trend": {
                "min_val": 1.23,
                "max_val": 15.67,
                "median_val": 8.45,
                "trend": "upward",
                "lags": [1, 2, 3, 4, 5]
            },
            "seasonal": {
                "min_val": -1.45,
                "max_val": 4.23,
                "median_val": 1.89,
                "trend": "upward",
                "lags": [7, 14, 21, 28, 35]
            }
        }
        
        prompts = {}
        for scenario_name, stats in scenarios.items():
            statistics = self.generate_statistics(**stats)
            prompt = self.generate_prompt(
                f"{scenario_name.title()} commodity forecasting",
                "forecast the next {pred_len} steps given the previous {seq_len} steps information",
                statistics,
                96,
                96
            )
            prompts[scenario_name] = prompt
        
        return prompts


def main():
    """Example usage of the prompt generator."""
    
    generator = TimeLlaMAPromptGenerator()
    
    print("=== Random Prompt ===")
    random_prompt = generator.generate_random_prompt()
    print(f"Dataset: {random_prompt['dataset_description']}")
    print(f"Task: {random_prompt['task_description']}")
    print(f"Prompt: {random_prompt['prompt']}")
    print()
    
    print("=== Mitsui Prompts ===")
    mitsui_prompts = generator.generate_mitsui_prompts(3)
    for i, prompt_data in enumerate(mitsui_prompts):
        print(f"Mitsui Prompt {i+1}: {prompt_data['prompt']}")
    print()
    
    print("=== Ablation Prompts ===")
    ablation_prompts = generator.generate_ablation_prompts()
    for variant, prompt in ablation_prompts.items():
        print(f"{variant}: {prompt}")
    print()
    
    print("=== Scenario Prompts ===")
    scenario_prompts = generator.generate_scenario_prompts()
    for scenario, prompt in scenario_prompts.items():
        print(f"{scenario}: {prompt}")


if __name__ == "__main__":
    main()
