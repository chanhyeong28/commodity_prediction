#!/bin/bash

# TimeLlaMA Mitsui Commodity Prediction Training Script

export CUDA_VISIBLE_DEVICES=0

model_name=TimeLlaMA

# Dataset configuration
root_path=./dataset/mitsui/
data_path=../kaggle/train.csv
labels_path=../kaggle/train_labels.csv

# Model configuration
llm_model=LLAMA
llm_layers=8  # As specified in TimeLlaMA paper: "maintaining the backbone model layers at 8 across all tasks"
d_model=512
d_ff=2048
n_heads=8
e_layers=2
dropout=0.1
embedding_type=linear

# Training configuration
train_epochs=10
batch_size=16
learning_rate=0.0001
patience=5

# Task configuration
task_type=long_term  # long_term or short_term

# Ablation study configuration (as per TimeLlaMA paper)
ablation_variant=full  # full, 1, 2, 3, 4, 5, 6
use_vanilla_lora=false
use_adalora=false
use_moelora=false

# DynaLoRA configuration
use_dynalora=true
dynalora_r_base=8
dynalora_n_experts=4
dynalora_dropout=0.0
dynalora_router_dropout=0.1
dynalora_load_balance_weight=0.01

echo "Starting TimeLlaMA training on Mitsui commodity prediction dataset..."

# Short-term forecasting (96 -> 96)
echo "Training for 96->96 prediction..."
python -u run_timellama.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --labels_path $labels_path \
  --model_id Mitsui_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Mitsui_96_96' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --embedding_type $embedding_type \
  --use_positional_emb \
  --use_channel_emb \
  --use_reprogramming \
  --freeze_llm \
  --task_type $task_type \
  --ablation_variant $ablation_variant \
  --use_vanilla_lora $use_vanilla_lora \
  --use_adalora $use_adalora \
  --use_moelora $use_moelora \
  --description "Mitsui commodity price forecasting" \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --patience $patience \
  --use_dynalora $use_dynalora \
  --dynalora_r_base $dynalora_r_base \
  --dynalora_n_experts $dynalora_n_experts \
  --dynalora_dropout $dynalora_dropout \
  --dynalora_router_dropout $dynalora_router_dropout \
  --dynalora_load_balance_weight $dynalora_load_balance_weight

# Medium-term forecasting (96 -> 192)
echo "Training for 96->192 prediction..."
python -u run_timellama.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --labels_path $labels_path \
  --model_id Mitsui_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Mitsui_96_192' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --embedding_type $embedding_type \
  --use_positional_emb \
  --use_channel_emb \
  --use_reprogramming \
  --freeze_llm \
  --task_type $task_type \
  --ablation_variant $ablation_variant \
  --use_vanilla_lora $use_vanilla_lora \
  --use_adalora $use_adalora \
  --use_moelora $use_moelora \
  --description "Mitsui commodity price forecasting" \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --patience $patience \
  --use_dynalora $use_dynalora \
  --dynalora_r_base $dynalora_r_base \
  --dynalora_n_experts $dynalora_n_experts \
  --dynalora_dropout $dynalora_dropout \
  --dynalora_router_dropout $dynalora_router_dropout \
  --dynalora_load_balance_weight $dynalora_load_balance_weight

# Long-term forecasting (96 -> 336)
echo "Training for 96->336 prediction..."
python -u run_timellama.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --labels_path $labels_path \
  --model_id Mitsui_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Mitsui_96_336' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --embedding_type $embedding_type \
  --use_positional_emb \
  --use_channel_emb \
  --use_reprogramming \
  --freeze_llm \
  --task_type $task_type \
  --ablation_variant $ablation_variant \
  --use_vanilla_lora $use_vanilla_lora \
  --use_adalora $use_adalora \
  --use_moelora $use_moelora \
  --description "Mitsui commodity price forecasting" \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --patience $patience \
  --use_dynalora $use_dynalora \
  --dynalora_r_base $dynalora_r_base \
  --dynalora_n_experts $dynalora_n_experts \
  --dynalora_dropout $dynalora_dropout \
  --dynalora_router_dropout $dynalora_router_dropout \
  --dynalora_load_balance_weight $dynalora_load_balance_weight

# Very long-term forecasting (96 -> 720)
echo "Training for 96->720 prediction..."
python -u run_timellama.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --labels_path $labels_path \
  --model_id Mitsui_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Mitsui_96_720' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --llm_model $llm_model \
  --llm_layers $llm_layers \
  --embedding_type $embedding_type \
  --use_positional_emb \
  --use_channel_emb \
  --use_reprogramming \
  --freeze_llm \
  --task_type $task_type \
  --ablation_variant $ablation_variant \
  --use_vanilla_lora $use_vanilla_lora \
  --use_adalora $use_adalora \
  --use_moelora $use_moelora \
  --description "Mitsui commodity price forecasting" \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --patience $patience \
  --use_dynalora $use_dynalora \
  --dynalora_r_base $dynalora_r_base \
  --dynalora_n_experts $dynalora_n_experts \
  --dynalora_dropout $dynalora_dropout \
  --dynalora_router_dropout $dynalora_router_dropout \
  --dynalora_load_balance_weight $dynalora_load_balance_weight

echo "All TimeLlaMA training experiments completed!"