# TimeLlaMA Prompts Directory

This directory contains mock text prompts and text data for the TimeLlaMA implementation.

## Files Overview

### 📄 `commodity_forecasting_prompts.txt`
Contains various prompt components and templates for commodity forecasting tasks:
- Dataset descriptions
- Task descriptions for different prediction horizons
- Input statistics templates
- Trend directions and market conditions

### 📄 `sample_prompts.json`
JSON file containing sample prompts with complete structure:
- 5 example prompts with full statistics
- Metadata about prompt format and structure
- Ready-to-use prompts for testing

### 📄 `ablation_prompts.txt`
Prompts specifically designed for ablation study variants:
- TimeLlaMA (Full) - Complete implementation
- TimeLlaMA-1 - No modality alignment
- TimeLlaMA-2 - Concatenate prompt as prefix
- TimeLlaMA-3 - Frozen LLM backbone
- TimeLlaMA-4 - Vanilla LoRA
- TimeLlaMA-5 - AdaLoRA
- TimeLlaMA-6 - MOELoRA

### 📄 `mitsui_specific_prompts.txt`
Prompts tailored for the Mitsui commodity prediction challenge:
- Different prediction horizons (96, 192, 336, 720 steps)
- Market condition specific prompts (bull, bear, sideways, volatile)
- Seasonal pattern prompts
- Economic indicator correlation prompts

### 📄 `prompt_templates.py`
Python module for generating TimeLlaMA prompts programmatically:
- `TimeLlaMAPromptGenerator` class
- Functions for generating random prompts
- Mitsui-specific prompt generation
- Ablation study prompt generation
- Scenario-based prompt generation

## TimeLlaMA Prompt Format

The TimeLlaMA prompt format follows this structure:

```
<|start_prompt|>Dataset description: {dataset_description} Task description: {task_description}; Input statistics: min value {min_val}, max value {max_val}, median value {median_val}, the trend of input is {trend}, top 5 lags are : {lags_list}<|<end_prompt|>
```

### Components:
- **Start/End Markers**: `<|start_prompt|>` and `<|<end_prompt|>`
- **Dataset Description**: Brief description of the forecasting task
- **Task Description**: Specific forecasting objective
- **Input Statistics**: Statistical properties of the input time series
  - min value, max value, median value
  - trend direction (upward, downward, sideways, volatile, etc.)
  - top 5 autocorrelation lags

## Usage Examples

### Using Pre-generated Prompts
```python
# Load sample prompts from JSON
import json
with open('sample_prompts.json', 'r') as f:
    prompts_data = json.load(f)
    
for prompt in prompts_data['prompts']:
    print(prompt['full_prompt'])
```

### Using Prompt Generator
```python
from prompt_templates import TimeLlaMAPromptGenerator

generator = TimeLlaMAPromptGenerator()

# Generate random prompt
random_prompt = generator.generate_random_prompt(pred_len=96, seq_len=96)
print(random_prompt['prompt'])

# Generate Mitsui-specific prompts
mitsui_prompts = generator.generate_mitsui_prompts(num_prompts=5)
for prompt_data in mitsui_prompts:
    print(prompt_data['prompt'])
```

### Using Ablation Prompts
```python
# Load ablation prompts
with open('ablation_prompts.txt', 'r') as f:
    ablation_content = f.read()
    
# Parse and use specific variants
lines = ablation_content.split('\n')
for line in lines:
    if line.startswith('<|start_prompt|>'):
        print(f"Ablation prompt: {line}")
```

## Integration with TimeLlaMA

These prompts are designed to work with the TimeLlaMA model's prompt generation system:

1. **Prompt Alignment**: Prompts are used for cross-attention alignment with time series tokens
2. **Ablation Studies**: Different prompt configurations test various model components
3. **Task Adaptation**: Prompts can be customized for different forecasting scenarios
4. **Evaluation**: Prompts help evaluate model performance across different conditions

## Customization

### Adding New Dataset Descriptions
Add new descriptions to the `dataset_descriptions` list in `prompt_templates.py`:
```python
self.dataset_descriptions.append("Your custom dataset description")
```

### Adding New Task Descriptions
Add new task descriptions with placeholders:
```python
self.task_descriptions.append("forecast the next {pred_len} steps given the previous {seq_len} steps information")
```

### Adding New Statistics
Extend the `generate_statistics` method to include additional statistical measures:
```python
statistics["new_metric"] = calculate_new_metric(data)
```

## Notes

- All prompts follow the exact format specified in the TimeLlaMA paper
- Statistics are generated to be realistic for commodity forecasting scenarios
- Prompts are designed to work with the Mitsui dataset structure
- The prompt generator can be extended for other time series forecasting tasks
- All prompts include proper start/end markers for tokenization
