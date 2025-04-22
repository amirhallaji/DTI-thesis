import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer, RobertaConfig
from rotary_embedding_torch import RotaryEmbedding


class GeGeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)




class FlashRobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads // 2
        self.head_dim = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.size()

        # Project hidden states to query, key, and value tensors
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        query = self.rotary_emb.rotate_queries_or_keys(query)
        key = self.rotary_emb.rotate_queries_or_keys(key)

        key = key.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=2)
        value = value.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=2)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)

        attention_output = self.o_proj(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        if head_mask is not None:
            attention_output = attention_output * head_mask

        if output_attentions:
            raise NotImplementedError("Output attentions is not supported with this implementation.")

        return (attention_output,)


class FlashRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GeGeLU()

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class FlashRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FlashRobertaAttention(config)
        self.intermediate = FlashRobertaIntermediate(config)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return (layer_output,)


class FlashRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta.encoder.layer = nn.ModuleList([FlashRobertaLayer(config) for _ in range(config.num_hidden_layers)])