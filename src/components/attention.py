import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention using RoPE positional encoding.

    Built on torch's scaled_dot_product_attention. Takes an explicit
    attention mask (rather than relying on SDPA's is_causal flag) so that
    packed training sequences can block attention across story boundaries
    while still applying causal masking within each story.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.rope = RotaryPositionalEmbeddings(self.head_dim, context_length)

    def forward(self, inp):
        x, attn_mask, positions = inp
        b, num_tokens, _ = x.shape

        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)

        keys = self.rope(keys, input_pos=positions)
        queries = self.rope(queries, input_pos=positions)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # combine the packed-sequence mask with an explicit causal mask, since
        # SDPA raises if both is_causal and attn_mask are set at the same time
        causal_mask = torch.tril(
            torch.ones(num_tokens, num_tokens, device=x.device)
        ).bool()
        attn_mask = attn_mask.view(b, 1, num_tokens, num_tokens)
        attn_mask = torch.logical_and(attn_mask, causal_mask).view(b, 1, num_tokens, num_tokens)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)
