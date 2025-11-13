from typing import Optional

import torch
from torch import nn
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.layers import Embedding, Linear

from transformers.configuration_utils import PretrainedConfig


class XiaoanConfig(PretrainedConfig):
    def __init__(self, *,
                 architecture,
                 vocab_size=32000,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_hidden_layers=6,
                 num_attention_heads=6,
                 **kwargs
                 ):
        super.__init__(self, **kwargs)
        self.architecture = architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

class XiaoanTransformer(nn.Module):
    def __init__(self, config):
        print(f'Xiaoan Transformer init')
        self.vocab_embedding = Embedding(config.vocab_size, config.hidden_size, 
                                         dtype = config.dtype)
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        hidden_states = self.vocab_embedding(input_ids)
        
        return hidden_states
        
class XiaoanModelForCausalLM(DecoderModelForCausalLM):
    config_class = XiaoanConfig
    
    def __init__(self, config):
        transformer = XiaoanTransformer(config)
        lm_head = Linear(config.hidden_size, config.vocab_size,
                         bias = False, dtype = config.dtype)
        super().__init__(config, transformer, lm_head)