from typing import Optional

import torch
from torch import nn
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

from configuration_mymodel import MyConfig


class MyAttention(Attention):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: Optional[int] = None):
        # Use model_config to initialize the Attention module
        super().__init__(...)


class MyDecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[MyConfig], layer_idx: int):
        super().__init__()
        # Use model_config to initialize the submodules
        self.input_layernorm = ...
        self.self_attn = MyAttention(model_config, layer_idx)
        self.post_attention_layernorm = ...
        self.mlp = ...

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        # Define the forward computation of a single decoder layer
        ...


class MyModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(model_config)
        # Use model_config to initialize the submodules
        self.embed_tokens = ...
        self.layers = nn.ModuleList([
            MyDecoderLayer(model_config, layer_idx) for layer_idx in range(model_config.pretrained_config.num_hidden_layers)
        ])

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):
        # Define the forward computation of the model
        ...


class MyModelForCausalLM(DecoderModelForCausalLM[MyModel, MyConfig]):
    def __init__(self, model_config: ModelConfig[MyConfig]):
        super().__init__(MyModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)