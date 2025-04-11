#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicity split the heads by adding a num_heads dimension
        # then unroll last dimension from (d_out) -> (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose from (b, num_tokens, num_heads,  head_dim)
        # to shape       (b, num_heads,  num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        # context_vec shape: (b, num_tokens, num_heads, head_dim)

        # combine heads where d_out = num_heads * head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)
        return context_vec


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
import torch.nn as nn


# this version uses the built-in scaled dot product attention
class MultiHeadAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicity split the heads by adding a num_heads dimension
        # then unroll last dimension from (d_out) -> (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose from (b, num_tokens, num_heads,  head_dim)
        # to shape       (b, num_heads,  num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, dropout_p=self.dropout, is_causal=True
        )
        # context_vec shape: (b, num_heads, num_tokens, head_dim)

        context_vec = context_vec.transpose(1, 2)
        # context_vec shape: (b, num_tokens, num_heads, head_dim)

        # combine heads where d_out = num_heads * head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)
        return context_vec


# In[1]:


import torch
import torch.nn as nn

# this version uses the built-in scaled dot product attention
# and uses a mask
class MultiHeadAttention_v3(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout


    def forward(self, inp):
        x, attn_mask = inp
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicity split the heads by adding a num_heads dimension
        # then unroll last dimension from (d_out) -> (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose from (b, num_tokens, num_heads,  head_dim)
        # to shape       (b, num_heads,  num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_mask = attn_mask.view(b, 1, num_tokens, num_tokens)
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, dropout_p=self.dropout, is_causal=True, attn_mask=attn_mask
        )
        # context_vec shape: (b, num_heads, num_tokens, head_dim)

        context_vec = context_vec.transpose(1, 2)
        # context_vec shape: (b, num_tokens, num_heads, head_dim)

        # combine heads where d_out = num_heads * head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)
        return context_vec


# In[ ]:


import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

# this version uses the built-in scaled dot product attention
# and uses a mask
class MultiHeadAttention_RoPE(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisble by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.rope = RotaryPositionalEmbeddings(self.head_dim, context_length)
        # self.register_buffer(
        #     "mask",
        #     torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # )


    def forward(self, inp):
        x, attn_mask, positions = inp
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicity split the heads by adding a num_heads dimension
        # then unroll last dimension from (d_out) -> (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # apply RoPE
        keys = self.rope(keys, input_pos=positions)
        queries = self.rope(queries, input_pos=positions)

        # transpose from (b, num_tokens, num_heads,  head_dim)
        # to shape       (b, num_heads,  num_tokens, head_dim)
        # print("keys shape", keys.shape)
        # print("positions shape", positions.shape)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)


        attn_mask = attn_mask.view(b, 1, num_tokens, num_tokens)

        # make causal mask
        causal_mask = torch.tril(
            torch.ones(num_tokens, num_tokens)
        ).to(x.device).bool()
        # use and to combine masks
        attn_mask = torch.logical_and(attn_mask, causal_mask)
        attn_mask = attn_mask.view(b, 1, num_tokens, num_tokens)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        # context_vec shape: (b, num_heads, num_tokens, head_dim)

        context_vec = context_vec.transpose(1, 2)
        # context_vec shape: (b, num_tokens, num_heads, head_dim)

        # combine heads where d_out = num_heads * head_dim
        # print("context_vec shape", context_vec.shape)
        # print("num_tokens", num_tokens)
        # print("b:", b)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, -1
        )

        context_vec = self.out_proj(context_vec)
        return context_vec


# In[ ]:





# In[215]:


# small sdpa test

def torch_sdpa(keys, queries, values, mask=None, is_causal=False):
    b, num_heads, num_tokens, head_dim = keys.shape

    if mask is not None:
        mask = mask.view(b, 1, num_tokens, num_tokens)

    attn_weights = torch.nn.functional.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=mask,
        is_causal=is_causal,
    )
    return attn_weights

def custom_sdpa(keys, queries, values, mask=None, is_causal=False):
    b, num_heads, num_tokens, head_dim = keys.shape

    if is_causal:
        causal_mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
        causal_mask = causal_mask.view(1, 1, num_tokens, num_tokens).expand(1, 1, -1, -1)
        # convert to bool
        causal_mask = causal_mask.bool()
        # print(causal_mask)

    attn_scores = queries @ keys.transpose(2, 3)
    attn_scores = attn_scores / head_dim**0.5
    if is_causal:
        attn_scores.masked_fill_(causal_mask, -torch.inf)
    if mask is not None:
        attn_scores.masked_fill_(~mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # this is necessary when all attn_scores for a specific value vector are -inf (due to masking)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    context_vec = (attn_weights @ values)
    # context_vec shape: (b, num_tokens, num_heads, head_dim)
    context_vec = context_vec.contiguous().view(
        b, num_heads, num_tokens, head_dim
    )
    return context_vec


# In[ ]:


# keys = torch.randn(2, 4, 5, 8)
# queries = torch.randn(2, 4, 5, 8)
# values = torch.randn(2, 4, 5, 8)

# torch_output = torch_sdpa(keys, queries, values)
# custom_output = custom_sdpa(keys, queries, values)
# print("implementation matches without mask?", torch.allclose(torch_output, custom_output, atol=1e-6))

# torch_causal_output = torch_sdpa(keys, queries, values, is_causal=True)
# custom_causal_output = custom_sdpa(keys, queries, values, is_causal=True)
# print("implementation matches with causal mask?", torch.allclose(torch_causal_output, custom_causal_output, atol=1e-6))

# mask = torch.randint(0, 2, (2, 1, 5, 5)).bool()
# # make sure no rows are all false
# while True:
#     if torch.any(mask.sum(dim=-1) == 0):
#         mask = torch.randint(0, 2, (2, 1, 5, 5)).bool()
#     else:
#         break



# torch_masked_output = torch_sdpa(keys, queries, values, mask=mask)
# custom_masked_output = custom_sdpa(keys, queries, values, mask=mask)
# print("implementation matches with mask?", torch.allclose(torch_masked_output, custom_masked_output, atol=1e-6))

# torch_causal_masked_output = torch_sdpa(keys, queries, values, mask=mask, is_causal=True)
# custom_causal_masked_output = custom_sdpa(keys, queries, values, mask=mask, is_causal=True)
# print("implementation matches with causal mask?", torch.allclose(torch_causal_masked_output, custom_causal_masked_output, atol=1e-6))


# In[ ]:




