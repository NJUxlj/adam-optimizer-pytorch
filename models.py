import time  
import numpy as np  
import gc  
import copy
import json
import torch  
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict  
from typing import Dict, List, Tuple, Optional  

from dataclasses import dataclass


from transformers import AutoTokenizer

MODEL_PATH = "D:\\models\\bert-base-chinese"


@dataclass
class TestModelConfig:
    vocab_size: int = 300
    hidden_size: int = 768
    num_layers: int = 2
    num_heads: int = 12
    ffn_hidden_size: int = 4 * hidden_size
    
    initializer_range: float = 0.02
    
    
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

class TestModelRMSNorm(nn.Module):
    '''
    RMS层归一化
    '''
    def __init__(self, hidden_size=768):
        super().__init__()

        self.w = nn.Parameter(torch.ones(hidden_size))
    
    
    def forward(self, x):
        '''
        x.shape = (batch_size, seq_len, hidden_size)
        
        论文公式：$归一化项 x_{RMS} = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$
        '''
        # 计算平方均值（RMS的核心）
        squared_mean = torch.mean(x**2, dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        
        # 计算RMS
        rms = torch.sqrt(squared_mean + 1e-6) # (batch_size, seq_len, 1)

        # 计算归一化
        norm = x / rms # (batch_size, seq_len, hidden_size)
        # 计算缩放
        scale = self.w * norm # (batch_size, seq_len, hidden_size)

        return scale
        



class TestModelLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, 4*hidden_size)
        self.act = F.silu
        self.ffn2 = nn.Linear(4*hidden_size, hidden_size)
        self.rms_norm = TestModelRMSNorm(hidden_size=hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self, x):
        '''
        x.shape = (batch_size, seq_len, hidden_size)
        '''
        # 前馈网络
        residual = x
        x = self.ffn1(x) # (batch_size, seq_len, 4*hidden_size)
        
        # print("ffn1.shape = ", x.shape, "residual.shape = ", residual.shape)
        x = self.act(x) # (batch_size, seq_len, 4*hidden_size)
        x = self.ffn2(x) # (batch_size, seq_len, hidden_size)
        # print("ffn2.shape = ", x.shape)
        # 残差连接
        x = x + residual # (batch_size, seq_len, hidden_size)

        # 层归一化
        x = self.rms_norm(x) # (batch_size, seq_len, hidden_size)

        # 丢弃
        x = self.dropout(x) # (batch_size, seq_len, hidden_size)
        
        return x



class TestPreTrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.config = TestModelConfig()
    def _init_weights(self, module):
        
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        


class TestModel(TestPreTrainedModel):
    '''
    测试模型，用于测试TorchOptimizerMonitor是否能够成功监控整个训练流程。
    '''
    def __init__(self, hidden_size=768, num_layers=2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.vocab_size = self.tokenizer.vocab_size
        
        self.embed = nn.Embedding(self.vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            TestModelLayer(hidden_size) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        '''
        x.shape = (batch_size, seq_len)
        '''
        x = self.embed(x) # (batch_size, seq_len, hidden_size)

        for layer in self.layers:
            x = layer(x) # (batch_size, seq_len, hidden_size)

        return x
        
        
        
        
class TestModelForCausalLM(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.model = TestModel(hidden_size)
        
        self.vocab_size = self.model.tokenizer.vocab_size
        
        self.lm_head = nn.Linear(hidden_size, self.vocab_size)
        
        self.loss_func = F.cross_entropy
        
    def forward(self, input_ids, attention_mask=None, labels = None):
        '''
        x.shape = (batch_size, seq_len)
        '''
        output = self.model(input_ids) # (batch_size, seq_len, hidden_size)
        logits = self.lm_head(output) # (batch_size, seq_len, vocab_size)

        
        loss = None
        
        
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.vocab_size), labels.view(-1))

        return (loss, logits) if loss is not None else (logits)
        
        