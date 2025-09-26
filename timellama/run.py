#!/usr/bin/env python3
"""
TimeLlaMA Training and Evaluation Script
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import warnings
import time

warnings.filterwarnings('ignore')

from timellama import TimeLlaMA
from timellama.utils.tools import EarlyStopping, adjust_learning_rate
from timellama.utils.metrics import metric

try:
    from timellama.data_provider import data_provider
except ImportError:
    print("Warning: data_provider not found.")
    data_provider = None

try:
    from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available.")
    TRANSFORMERS_AVAILABLE = False


class Exp_TimeLlaMA:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        if self.args.llm_model == 'LLAMA':
            # Use LlaMA-2 7B as specified in the TimeLlaMA paper
            try:
                # Try official LlaMA-2 7B model first
                llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
                llama_config.num_hidden_layers = self.args.llm_layers
                
                llm_model = LlamaModel.from_pretrained('meta-llama/Llama-2-7b-hf', config=llama_config)
                tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
                print("Using official LlaMA-2 7B model as specified in TimeLlaMA paper")
            except Exception as e:
                print(f"Failed to load official LlaMA-2 model: {e}")
                print("Falling back to alternative LlaMA model...")
                # Fallback to alternative model
                llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                llama_config.num_hidden_layers = self.args.llm_layers
                
                llm_model = LlamaModel.from_pretrained('huggyllama/llama-7b', config=llama_config)
                tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
                print("Using fallback LlaMA model (note: this may not match paper exactly)")
        
        elif self.args.llm_model == 'GPT2':
            gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            gpt2_config.num_hidden_layers = self.args.llm_layers
            
            llm_model = GPT2Model.from_pretrained('openai-community/gpt2', config=gpt2_config)
            tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        
        else:
            raise ValueError(f"Unsupported LLM model: {self.args.llm_model}")
        
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = '[PAD]'
        
        # Configure ablation variant settings
        ablation_config = self._configure_ablation_variant()
        
        model = TimeLlaMA(
            llm_model=llm_model,
            tokenizer=tokenizer,
            d_model=self.args.d_model,
            lookback=self.args.seq_len,
            pred_len=self.args.pred_len,
            num_channels=self.args.enc_in,
            num_heads=self.args.n_heads,
            d_ff=self.args.d_ff,
            embedding_type=self.args.embedding_type,
            use_positional_emb=self.args.use_positional_emb,
            use_channel_emb=self.args.use_channel_emb,
            dropout=self.args.dropout,
            head_dropout=self.args.head_dropout,
            use_reprogramming=ablation_config['use_reprogramming'],
            description=self.args.description,
            freeze_llm=ablation_config['freeze_llm'],
            # DynaLoRA parameters
            use_dynalora=ablation_config['use_dynalora'],
            dynalora_r_base=self.args.dynalora_r_base,
            dynalora_n_experts=self.args.dynalora_n_experts,
            dynalora_dropout=self.args.dynalora_dropout,
            dynalora_router_dropout=self.args.dynalora_router_dropout,
            # Ablation variant parameters
            ablation_variant=self.args.ablation_variant,
            use_vanilla_lora=self.args.use_vanilla_lora,
            use_adalora=self.args.use_adalora,
            use_moelora=self.args.use_moelora
        )
        
        return model

    def _configure_ablation_variant(self):
        """
        Configure model settings based on ablation variant as per TimeLlaMA paper.
        
        Returns:
            dict: Configuration settings for the specified ablation variant
        """
        variant = self.args.ablation_variant
        
        if variant == 'full':
            # Full Time-LLaMA (default)
            return {
                'use_reprogramming': True,
                'freeze_llm': False,  # Fine-tune with DynaLoRA
                'use_dynalora': True,
                'use_prompt_alignment': True
            }
        elif variant == '1':
            # Time-LLaMA-1: Remove modality alignment module
            return {
                'use_reprogramming': False,
                'freeze_llm': False,
                'use_dynalora': True,
                'use_prompt_alignment': False
            }
        elif variant == '2':
            # Time-LLaMA-2: Concatenate text prompt as prefix
            return {
                'use_reprogramming': True,
                'freeze_llm': False,
                'use_dynalora': True,
                'use_prompt_alignment': True,
                'concatenate_prompt': True
            }
        elif variant == '3':
            # Time-LLaMA-3: Keep LLM backbone entirely frozen
            return {
                'use_reprogramming': True,
                'freeze_llm': True,
                'use_dynalora': False,
                'use_prompt_alignment': True
            }
        elif variant == '4':
            # Time-LLaMA-4: Substitute DynaLoRA with vanilla LoRA
            return {
                'use_reprogramming': True,
                'freeze_llm': False,
                'use_dynalora': False,
                'use_vanilla_lora': True,
                'use_prompt_alignment': True
            }
        elif variant == '5':
            # Time-LLaMA-5: Substitute DynaLoRA with AdaLoRA
            return {
                'use_reprogramming': True,
                'freeze_llm': False,
                'use_dynalora': False,
                'use_adalora': True,
                'use_prompt_alignment': True
            }
        elif variant == '6':
            # Time-LLaMA-6: Substitute DynaLoRA with MOELoRA
            return {
                'use_reprogramming': True,
                'freeze_llm': False,
                'use_dynalora': False,
                'use_moelora': True,
                'use_prompt_alignment': True
            }
        else:
            raise ValueError(f"Unknown ablation variant: {variant}")

    def _get_data(self, flag):
        if data_provider is None:
            raise ImportError("Data provider not available")
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        Select loss function based on task type as per TimeLlaMA paper:
        - MSE for long-term forecasting tasks
        - SMAPE for short-term forecasting tasks
        """
        if hasattr(self.args, 'task_type') and self.args.task_type == 'short_term':
            from timellama.utils.losses import smape_loss
            return smape_loss()
        else:
            # Default to MSE for long-term forecasting
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            # Reset expert usage statistics at the beginning of each epoch
            if hasattr(self.model, 'reset_expert_usage_stats'):
                self.model.reset_expert_usage_stats()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                
                # Add DynaLoRA regularization loss
                if hasattr(self.model, 'get_dynalora_regularization_loss'):
                    dynalora_loss = self.model.get_dynalora_regularization_loss()
                    loss = loss + dynalora_loss
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    
                    # Log expert usage statistics every 100 steps
                    if hasattr(self.model, 'get_expert_usage_stats'):
                        expert_stats = self.model.get_expert_usage_stats()
                        if expert_stats:
                            print(f"\tExpert Usage Stats:")
                            for layer_name, stats in expert_stats.items():
                                print(f"\t  {layer_name}: {stats}")

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Use appropriate metrics based on task type as per TimeLlaMA paper
        if hasattr(self.args, 'task_type') and self.args.task_type == 'short_term':
            from timellama.utils.metrics import short_term_metrics
            smape, mase, owa = short_term_metrics(preds, trues)
            print('SMAPE:{}, MASE:{}, OWA:{}'.format(smape, mase, owa))
            np.save(folder_path + 'metrics.npy', np.array([smape, mase, owa]))
        else:
            # Long-term forecasting metrics
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('MSE:{}, MAE:{}'.format(mse, mae))
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TimeLlaMA')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='TimeLlaMA')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='/Users/minkeychang/commodity_prediction/kaggle/train.csv')
    parser.add_argument('--labels_path', type=str, default='/Users/minkeychang/commodity_prediction/kaggle/train_labels.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--max_targets', type=int, default=50, help='maximum number of targets to use (for memory efficiency)')
    parser.add_argument('--task_type', type=str, default='long_term', choices=['long_term', 'short_term'], 
                        help='Task type: long_term (MSE loss) or short_term (SMAPE loss)')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # model define
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    # TimeLlaMA specific
    parser.add_argument('--llm_model', type=str, default='LLAMA')
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--embedding_type', type=str, default='linear')
    parser.add_argument('--use_positional_emb', action='store_true', default=True)
    parser.add_argument('--use_channel_emb', action='store_true', default=True)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--use_reprogramming', action='store_true', default=True)
    parser.add_argument('--description', type=str, default='Time series forecasting task')
    parser.add_argument('--freeze_llm', action='store_true', default=True)
    
    # DynaLoRA specific
    parser.add_argument('--use_dynalora', action='store_true', default=False)
    parser.add_argument('--dynalora_r_base', type=int, default=8)
    parser.add_argument('--dynalora_n_experts', type=int, default=4)
    parser.add_argument('--dynalora_dropout', type=float, default=0.0)
    parser.add_argument('--dynalora_router_dropout', type=float, default=0.1)
    parser.add_argument('--dynalora_load_balance_weight', type=float, default=0.01)
    
    # Ablation study variants as per TimeLlaMA paper
    parser.add_argument('--ablation_variant', type=str, default='full', 
                        choices=['full', '1', '2', '3', '4', '5', '6'],
                        help='Ablation variant: full=Time-LLaMA, 1-6=Time-LLaMA-1 to Time-LLaMA-6')
    parser.add_argument('--use_vanilla_lora', action='store_true', default=False,
                        help='Use vanilla LoRA instead of DynaLoRA (Time-LLaMA-4)')
    parser.add_argument('--use_adalora', action='store_true', default=False,
                        help='Use AdaLoRA instead of DynaLoRA (Time-LLaMA-5)')
    parser.add_argument('--use_moelora', action='store_true', default=False,
                        help='Use MOELoRA instead of DynaLoRA (Time-LLaMA-6)')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--lradj', type=str, default='type1')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    # fix random seed
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    if args.is_training:
        for ii in range(1):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
                args.e_layers, args.d_layers, args.d_ff, args.des, ii)

            exp = Exp_TimeLlaMA(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
    else:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.des, 0)

        exp = Exp_TimeLlaMA(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
