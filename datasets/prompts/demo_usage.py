#!/usr/bin/env python3
"""
Demo script showing how to use the TimeLlaMA prompts.

This script demonstrates various ways to load and use the mock prompts
for TimeLlaMA training and evaluation.
"""

import json
import random
from prompt_templates import TimeLlaMAPromptGenerator


def demo_json_prompts():
    """Demonstrate loading prompts from JSON file."""
    print("=== Loading Prompts from JSON ===")
    
    try:
        with open('sample_prompts.json', 'r') as f:
            prompts_data = json.load(f)
        
        print(f"Loaded {len(prompts_data['prompts'])} prompts")
        
        # Show first prompt
        first_prompt = prompts_data['prompts'][0]
        print(f"\nFirst prompt:")
        print(f"ID: {first_prompt['id']}")
        print(f"Dataset: {first_prompt['dataset_description']}")
        print(f"Task: {first_prompt['task_description']}")
        print(f"Statistics: {first_prompt['input_statistics']}")
        print(f"Full prompt: {first_prompt['full_prompt']}")
        
        return prompts_data['prompts']
        
    except FileNotFoundError:
        print("sample_prompts.json not found")
        return []


def demo_prompt_generator():
    """Demonstrate using the prompt generator."""
    print("\n=== Using Prompt Generator ===")
    
    generator = TimeLlaMAPromptGenerator()
    
    # Generate random prompt
    print("Random prompt:")
    random_prompt = generator.generate_random_prompt(pred_len=96, seq_len=96)
    print(f"Dataset: {random_prompt['dataset_description']}")
    print(f"Task: {random_prompt['task_description']}")
    print(f"Statistics: {random_prompt['statistics']}")
    print(f"Prompt: {random_prompt['prompt']}")
    
    # Generate Mitsui-specific prompts
    print("\nMitsui-specific prompts:")
    mitsui_prompts = generator.generate_mitsui_prompts(num_prompts=3)
    for i, prompt_data in enumerate(mitsui_prompts):
        print(f"\nMitsui Prompt {i+1}:")
        print(f"Prediction length: {prompt_data['pred_len']}")
        print(f"Prompt: {prompt_data['prompt']}")
    
    return generator


def demo_ablation_prompts():
    """Demonstrate ablation study prompts."""
    print("\n=== Ablation Study Prompts ===")
    
    generator = TimeLlaMAPromptGenerator()
    ablation_prompts = generator.generate_ablation_prompts()
    
    for variant, prompt in ablation_prompts.items():
        print(f"\n{variant.upper()}:")
        print(f"{prompt}")


def demo_scenario_prompts():
    """Demonstrate scenario-based prompts."""
    print("\n=== Scenario-Based Prompts ===")
    
    generator = TimeLlaMAPromptGenerator()
    scenario_prompts = generator.generate_scenario_prompts()
    
    for scenario, prompt in scenario_prompts.items():
        print(f"\n{scenario.upper()}:")
        print(f"{prompt}")


def demo_custom_prompts():
    """Demonstrate creating custom prompts."""
    print("\n=== Custom Prompts ===")
    
    generator = TimeLlaMAPromptGenerator()
    
    # Custom statistics for a specific scenario
    custom_stats = generator.generate_statistics(
        min_val=-5.0,
        max_val=10.0,
        median_val=2.5,
        trend="upward",
        lags=[1, 7, 14, 21, 28]
    )
    
    # Generate custom prompt
    custom_prompt = generator.generate_prompt(
        dataset_description="Custom commodity forecasting task",
        task_description="forecast the next {pred_len} steps given the previous {seq_len} steps information",
        statistics=custom_stats,
        pred_len=192,
        seq_len=96
    )
    
    print("Custom prompt with specific parameters:")
    print(f"Statistics: {custom_stats}")
    print(f"Prompt: {custom_prompt}")


def demo_prompt_analysis():
    """Demonstrate analyzing prompt characteristics."""
    print("\n=== Prompt Analysis ===")
    
    generator = TimeLlaMAPromptGenerator()
    
    # Generate multiple prompts and analyze
    prompts = []
    for i in range(10):
        prompt_data = generator.generate_random_prompt()
        prompts.append(prompt_data)
    
    # Analyze trends
    trends = [p['statistics']['trend'] for p in prompts]
    trend_counts = {}
    for trend in trends:
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
    
    print("Trend distribution in 10 random prompts:")
    for trend, count in trend_counts.items():
        print(f"  {trend}: {count}")
    
    # Analyze volatility
    volatilities = [p['statistics']['volatility'] for p in prompts]
    vol_counts = {}
    for vol in volatilities:
        vol_counts[vol] = vol_counts.get(vol, 0) + 1
    
    print("\nVolatility distribution in 10 random prompts:")
    for vol, count in vol_counts.items():
        print(f"  {vol}: {count}")


def demo_integration_example():
    """Demonstrate how prompts would be integrated with TimeLlaMA."""
    print("\n=== Integration Example ===")
    
    generator = TimeLlaMAPromptGenerator()
    
    # Simulate TimeLlaMA prompt generation process
    print("Simulating TimeLlaMA prompt generation:")
    
    # Mock time series data (normally this would come from actual data)
    mock_data_stats = {
        "min_val": -2.34,
        "max_val": 5.67,
        "median_val": 1.23,
        "trend": "upward",
        "lags": [1, 7, 14, 21, 30]
    }
    
    # Generate statistics (normally calculated from actual data)
    statistics = generator.generate_statistics(**mock_data_stats)
    
    # Generate prompt (this is what TimeLlaMA does)
    prompt = generator.generate_prompt(
        dataset_description="Mitsui commodity price prediction challenge",
        task_description="forecast the next {pred_len} steps given the previous {seq_len} steps information",
        statistics=statistics,
        pred_len=96,
        seq_len=96
    )
    
    print(f"Generated prompt: {prompt}")
    
    # This prompt would then be used for:
    # 1. Tokenization by the LLM tokenizer
    # 2. Embedding generation
    # 3. Cross-attention with time series tokens
    # 4. Alignment in the TimeLlaMA model
    
    print("\nThis prompt would be used for:")
    print("1. Tokenization by LLM tokenizer")
    print("2. Embedding generation")
    print("3. Cross-attention with time series tokens")
    print("4. Alignment in TimeLlaMA model")


def main():
    """Run all demonstrations."""
    print("TimeLlaMA Prompts Demo")
    print("=" * 50)
    
    # Demo 1: JSON prompts
    json_prompts = demo_json_prompts()
    
    # Demo 2: Prompt generator
    generator = demo_prompt_generator()
    
    # Demo 3: Ablation prompts
    demo_ablation_prompts()
    
    # Demo 4: Scenario prompts
    demo_scenario_prompts()
    
    # Demo 5: Custom prompts
    demo_custom_prompts()
    
    # Demo 6: Prompt analysis
    demo_prompt_analysis()
    
    # Demo 7: Integration example
    demo_integration_example()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo use these prompts in your TimeLlaMA implementation:")
    print("1. Load prompts from JSON files or generate them programmatically")
    print("2. Use the TimeLlaMA model's generate_prompt() method")
    print("3. Pass prompts to build_prompt_embeddings() for alignment")
    print("4. Use in cross-attention with time series tokens")


if __name__ == "__main__":
    main()
