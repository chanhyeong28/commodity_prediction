#!/bin/bash

# TimeLlaMA Ablation Study Script
# Implements the 6 ablation variants from the TimeLlaMA paper

# Dataset configuration
root_path=./dataset/mitsui/
data_path=../kaggle/train.csv
labels_path=../kaggle/train_labels.csv
max_targets=50

# Model configuration
llm_model=LLAMA
llm_layers=8  # As specified in TimeLlaMA paper
d_model=512
d_ff=2048
n_heads=8
e_layers=2
dropout=0.1

# Training configuration
train_epochs=10
batch_size=16
learning_rate=0.0001
patience=5

# Task configuration
task_type=long_term

# DynaLoRA configuration
use_dynalora=true
dynalora_r_base=8
dynalora_n_experts=4
dynalora_dropout=0.0
dynalora_router_dropout=0.1
dynalora_load_balance_weight=0.01

echo "Starting TimeLlaMA Ablation Study on Mitsui commodity prediction dataset..."

# Function to run experiment
run_experiment() {
    local variant=$1
    local variant_name=$2
    local description=$3
    
    echo "Running $variant_name: $description"
    
    python -u run.py \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --labels_path $labels_path \
      --model_id Mitsui_ablation_${variant} \
      --model TimeLlaMA \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 96 \
      --e_layers $e_layers \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des "Mitsui_ablation_${variant}" \
      --d_model $d_model \
      --d_ff $d_ff \
      --n_heads $n_heads \
      --dropout $dropout \
      --llm_model $llm_model \
      --llm_layers $llm_layers \
      --use_positional_emb \
      --use_channel_emb \
      --use_reprogramming \
      --freeze_llm \
      --task_type $task_type \
      --ablation_variant $variant \
      --use_vanilla_lora false \
      --use_adalora false \
      --use_moelora false \
      --description "$description" \
      --train_epochs $train_epochs \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --patience $patience \
      --itr 1
    
    echo "Completed $variant_name"
    echo "----------------------------------------"
}

# Run all ablation variants
echo "=== TimeLlaMA Ablation Study ==="

# Full TimeLlaMA (baseline)
run_experiment "full" "TimeLlaMA (Full)" "Full TimeLlaMA with all components"

# TimeLlaMA-1: Remove modality alignment module
run_experiment "1" "TimeLlaMA-1" "Remove modality alignment module, direct feed to LLM"

# TimeLlaMA-2: Concatenate text prompt as prefix
run_experiment "2" "TimeLlaMA-2" "Concatenate text prompt to left of time-series tokens"

# TimeLlaMA-3: Keep LLM backbone entirely frozen
run_experiment "3" "TimeLlaMA-3" "Keep LLM backbone entirely frozen"

# TimeLlaMA-4: Substitute DynaLoRA with vanilla LoRA
run_experiment "4" "TimeLlaMA-4" "Substitute DynaLoRA with vanilla LoRA"

# TimeLlaMA-5: Substitute DynaLoRA with AdaLoRA
run_experiment "5" "TimeLlaMA-5" "Substitute DynaLoRA with AdaLoRA"

# TimeLlaMA-6: Substitute DynaLoRA with MOELoRA
run_experiment "6" "TimeLlaMA-6" "Substitute DynaLoRA with MOELoRA"

echo "=== Ablation Study Complete ==="
echo "Results saved in ./results/ directory"
echo "Compare performance across variants to validate TimeLlaMA components"
