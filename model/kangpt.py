from typing import Optional, Tuple, Union
import torch
from torch import nn

from .gpt2 import GPT2Attention
from .kan import KAN


class KANGPTBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.kan = KAN(config.kan_layers_hidden, config.kan_grid_size, config.kan_spline_order)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        feed_forward_hidden_states = self.kan(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    

class KANGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([KANGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
        causal_mask = self._create_causal_mask(config.max_position_embeddings)
        self.register_buffer('causal_mask', causal_mask, persistent=False)
    
    def _create_causal_mask(self, maxlen):
        causal_mask = torch.ones(maxlen, maxlen).tril() == 0
        causal_mask = torch.where(causal_mask, -float('inf'), 0)
        return causal_mask

    def _get_causal_mask(self, n_positions):
        return self.causal_mask[:n_positions, :n_positions]

    def forward(self, input_ids: Optional[torch.LongTensor]):        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe.weight[:input_ids.size(1)]
        causal_mask = self._get_causal_mask(input_ids.size(1))
        
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        for block in self.h:
            outputs = block(hidden_states, attention_mask=causal_mask)
            hidden_states = outputs[0]

 
        last_hidden_states = self.ln_f(hidden_states)
        return last_hidden_states


class KANGPTLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = KANGPT(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor]
    ):
        outputs = self.transformer(input_ids, attention_mask)
        logits = self.lm_head(outputs)
        return logits
