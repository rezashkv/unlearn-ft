from typing import Optional
import torch
import torch.nn as nn
from diffusers.utils import deprecate
from pdm.utils.estimation_utils import hard_concrete

from .activations import GatedGELU, GatedGEGLU, GatedApproximateGELU, GatedSwiGLU
from diffusers.models.attention_processor import (
    Attention,
    FluxAttnProcessor2_0,
    FluxSingleAttnProcessor2_0
)

from .gates import WidthGate
import torch.nn.functional as F


class GatedFeedForward(nn.Module):
    r"""
    A gated feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
        gate_width (`int`, *optional*, defaults to 32): The width of the gate.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
            final_dropout: bool = False,
            inner_dim=None,
            bias: bool = True,
            gate_width: int = 32,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GatedGELU(dim, inner_dim, bias=bias, gate_width=gate_width)
        if activation_fn == "gelu-approximate":
            act_fn = GatedGELU(dim, inner_dim, approximate="tanh", bias=bias, gate_width=gate_width)
        elif activation_fn == "geglu":
            act_fn = GatedGEGLU(dim, inner_dim, bias=bias, gate_width=gate_width)
        elif activation_fn == "geglu-approximate":
            act_fn = GatedApproximateGELU(dim, inner_dim, bias=bias, gate_width=gate_width)
        elif activation_fn == "swiglu":
            act_fn = GatedSwiGLU(dim, inner_dim, bias=bias, gate_width=gate_width)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

        self.prunable_macs, self.total_macs = 0., 0.

    def calc_macs(self):
        if self.total_macs == 0. or self.prunable_macs == 0.:
            self.total_macs, self.prunable_macs = 0., 0.
            # act
            self.total_macs += self.net[0].proj.__macs__
            self.prunable_macs += self.net[0].proj.__macs__

            # Linear
            self.total_macs += self.net[2].__macs__
            self.prunable_macs += self.net[2].__macs__

        hard_width_gate = hard_concrete(self.net[0].gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ratio * self.prunable_macs,
                "cur_total_macs": ratio.detach() * self.prunable_macs + (self.total_macs - self.prunable_macs)}



    @torch.no_grad()
    def prune(self):
        gate_hard = self.net[0].prune_gate()
        linear_cls = self.net[2].__class__
        linear = linear_cls(self.net[0].proj.out_features, self.net[2].out_features, bias=self.net[2].bias is not None)
        linear.weight.data = self.net[2].weight.data[:, gate_hard.bool()]
        if self.net[2].bias is not None:
            linear.bias.data = self.net[2].bias.data
        self.net[2] = linear

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GatedFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply the gate
        query = attn.gate(query)
        key = attn.gate(key)
        value = attn.gate(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        # Apply the gate
        encoder_hidden_states_query_proj = attn.gate(encoder_hidden_states_query_proj)
        encoder_hidden_states_key_proj = attn.gate(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.gate(encoder_hidden_states_value_proj)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query, key = apply_rope(query, key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


class GatedFluxSingleAttnProcessor2_0(FluxSingleAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply the gate
        query = attn.gate(query)
        key = attn.gate(key)
        value = attn.gate(value)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:

            query, key = apply_rope(query, key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


class GatedAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = WidthGate(self.heads)
        self.prunable_macs, self.total_macs = 0., 0.
        self.pruned = False

    def calc_macs(self):
        assert ((self.total_macs != 0.) and (self.prunable_macs != 0.))
        hard_width_gate = hard_concrete(self.gate.gate_f)
        ratio = hard_width_gate.sum(dim=1, keepdim=True) / hard_width_gate.shape[1]
        return {"prunable_macs": self.prunable_macs,
                "total_macs": self.total_macs,
                "cur_prunable_macs": ratio * self.prunable_macs,
                "cur_total_macs": ratio.detach() * self.prunable_macs + (self.total_macs - self.prunable_macs)}

    def get_prunable_macs(self):
        return [self.prunable_macs]

    @torch.no_grad()
    def prune(self):
        # todo add support for added_kv_proj and norms
        def prune_linear(layer, gate_hard, out=False):
            num_new_heads = gate_hard.sum().int().item()
            assert num_new_heads > 0
            linear_cls = layer.__class__
            if out:
                head_dim = layer.in_features // self.heads
                linear = linear_cls(num_new_heads * head_dim, layer.out_features, bias=layer.bias is not None)
                orig_linear_weight = layer.weight.data.view(layer.out_features, self.heads, head_dim)
                new_linear_weight = orig_linear_weight[:, gate_hard.bool(), :].view(layer.out_features, -1)
            else:
                head_dim = layer.out_features // self.heads
                linear = linear_cls(layer.in_features, num_new_heads * head_dim, bias=layer.bias is not None)
                orig_linear_weight = layer.weight.data.view(self.heads, head_dim, layer.in_features)
                new_linear_weight = orig_linear_weight[gate_hard.bool(), :, :].view(-1, layer.in_features)

            linear.weight.data = new_linear_weight
            if layer.bias is not None:
                if out:
                    linear.bias.data = layer.bias.data
                else:
                    orig_linear_bias = layer.bias.data.view(self.heads, head_dim)
                    new_linear_bias = orig_linear_bias[gate_hard.bool(), :].view(-1)
                    linear.bias.data = new_linear_bias
            return linear

        assert self.gate.gate_f.shape[0] == 1, "Pruning is only supported for single batch size"
        gate_h = hard_concrete(self.gate.gate_f)[0]
        self.to_q = prune_linear(self.to_q, gate_h)
        self.to_k = prune_linear(self.to_k, gate_h)
        self.to_v = prune_linear(self.to_v, gate_h)
        self.to_out[0] = prune_linear(self.to_out[0], gate_h, out=True)
        self.heads = gate_h.sum().int().item()
        self.pruned = True