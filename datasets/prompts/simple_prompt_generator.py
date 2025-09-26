#!/usr/bin/env python3
"""
Simple prompt generator for TimeLlaMA (no external dependencies).

This is a simplified version that doesn't require numpy or other external libraries.
"""

import random
from typing import Dict, List, Any, Optional


class SimpleTimeLlaMAPromptGenerator:
    """Simple generator for TimeLlaMA-style prompts (no external dependencies)."""
    
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


def main():
    """Example usage of the simple prompt generator."""
    
    generator = SimpleTimeLlaMAPromptGenerator()
    
    print("=== Simple TimeLlaMA Prompt Generator ===")
    
    # Generate random prompt
    print("\nRandom prompt:")
    random_prompt = generator.generate_random_prompt()
    print(f"Dataset: {random_prompt['dataset_description']}")
    print(f"Task: {random_prompt['task_description']}")
    print(f"Statistics: {random_prompt['statistics']}")
    print(f"Prompt: {random_prompt['prompt']}")
    
    # Generate Mitsui-specific prompts
    print("\nMitsui-specific prompts:")
    mitsui_prompts = generator.generate_mitsui_prompts(3)
    for i, prompt_data in enumerate(mitsui_prompts):
        print(f"\nMitsui Prompt {i+1}:")
        print(f"Prediction length: {prompt_data['pred_len']}")
        print(f"Prompt: {prompt_data['prompt']}")
    
    print("\n✅ Simple prompt generator working correctly!")


if __name__ == "__main__":
    main()
